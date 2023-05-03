import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import cv2
from Renderer.model import *
from DRL.rpm import rpm
from DRL.actor import *
from DRL.critic import *
from DRL.wgan import *
from utils.util import *
from DRL.content_loss import *
from DRL.vgg import *
from DRL.gram_matrix import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

coord = torch.zeros([1, 2, 128, 128])
for i in range(128):
    for j in range(128):
        coord[0, 0, i, j] = i / 127.
        coord[0, 1, i, j] = j / 127.
coord = coord.to(device)

criterion = nn.MSELoss()

Decoder = FCN()
Decoder.load_state_dict(torch.load('./renderer_constrained.pkl'))
vgg = VGG()
vgg.load_state_dict(torch.load('./vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()

def decode(x, canvas): # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    stroke = 1 - Decoder(x[:, :10])
    stroke = stroke.view(-1, 128, 128, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, 128, 128)
    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas

def cal_trans(s, t):
    return (s.transpose(0, 3) * t).transpose(0, 3)

def cal_content_loss(canvas0, canvas1, target):
    return ((canvas0 - target) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - target) ** 2).mean(1).mean(1).mean(1)

def cal_style_loss(canvas0, canvas1, target):
    out_0 = GramMatrix()(canvas0)
    out_1 = GramMatrix()(canvas1)
    return ((out_0 - target) ** 2).mean(1).mean(1) - ((out_1 - target) ** 2).mean(1).mean(1)

def cal_perceptual_style_reward(canvas0, canvas1, target):
    #style_targets = [GramMatrix()(A).detach() for A in vgg(target, style_layers)]
    content_targets = [A.detach() for A in vgg(target, content_layers)]
    targets = style_targets + content_targets
    out_canvas_0 = vgg(canvas0, loss_layers)
    out_canvas_1 = vgg(canvas1, loss_layers)
    # perceptual loss (canvas0, target)
    #layer_losses_0 = [loss_fns[a](A, content_targets[a]) for a,A in enumerate(out_canvas_0)]
    #loss_0 = sum(layer_losses_0)
    # perceptual loss (canvas1, target)
    #layer_losses_1 = [loss_fns[a](A, content_targets[a]) for a,A in enumerate(out_canvas_1)]
    layer_reward = [loss_fns[i](out_canvas_0[i],out_canvas_1[i], targets[i]) for i in range(len(loss_layers))]
    style_reward = layer_reward[0]
    for i in range(1, len(style_layers)):
      style_reward += layer_reward[i]
    style_reward /= style_scale
    content_reward = layer_reward[-1]
    content_reward = content_mask_l1_reward(canvas0, canvas1, target)[0] / content_scale
    reward = style_weight * style_reward + content_weight * content_reward
    #perceptual_reward = ((out_canvas_0[0] - content_targets[0]) ** 2).mean(1).mean(1).mean(1) - ((out_canvas_1[0] - content_targets[0]) ** 2).mean(1).mean(1).mean(1)
    return reward

def cml1_style_reward_style_dataset(canvas0, canvas1, gt):
    style_targets = [GramMatrix()(A).detach() for A in vgg(gt, style_layers)]
    content_targets = [A.detach() for A in vgg(gt, content_layers)]
    targets = style_targets + content_targets
    out_canvas_0 = vgg(canvas0, loss_layers)
    out_canvas_1 = vgg(canvas1, loss_layers)
    layer_reward = [weights[i] * loss_fns[i](out_canvas_0[i],out_canvas_1[i], targets[i]) for i in range(len(loss_layers))]
    style_reward = layer_reward[0]
    for i in range(1, len(style_layers)):
      style_reward += layer_reward[i]
    style_reward /= style_scale
    content_reward, mask = content_mask_l1_reward(canvas0, canvas1, gt)
    content_reward /= content_scale
    # print('content reward')
    # print(content_reward)
    # print(style_reward)
    reward = style_weight * style_reward + content_weight * content_reward

    return reward, mask

def cml1_style_reward_style_img(canvas0, canvas1, gt):
    targets = style_targets_img
    out_canvas_0 = vgg(canvas0, style_layers)
    out_canvas_1 = vgg(canvas1, style_layers)
    # perceptual loss (canvas0, target)
    #layer_losses_0 = [loss_fns[a](A, content_targets[a]) for a,A in enumerate(out_canvas_0)]
    #loss_0 = sum(layer_losses_0)
    # perceptual loss (canvas1, target)
    #layer_losses_1 = [loss_fns[a](A, content_targets[a]) for a,A in enumerate(out_canvas_1)]
    layer_reward = [weights[i] * loss_fns[i](out_canvas_0[i],out_canvas_1[i], targets[i]) for i in range(len(style_layers))]
    style_reward = layer_reward[0]
    for i in range(1, len(style_layers)):
      style_reward += layer_reward[i]
    style_reward /= style_scale
    content_reward, mask = content_mask_l1_reward(canvas0, canvas1, gt)
    content_reward /= content_scale
    # print()
    # print(content_reward.sum())
    # print(style_reward.sum())
    reward = style_weight * style_reward + content_weight * content_reward
    return reward, mask

def content_mask_l1_reward(canvas0, canvas1, gt):
    mask = get_l2_mask(gt).to(torch.float)
    canvas0 = canvas0.to(torch.float)
    canvas1 = canvas1.to(torch.float)
    gt = gt.to(torch.float)
    clip = 0.2

    l1_0 = torch.abs(canvas0-gt)
    l1_0[l1_0 > clip] = clip

    l1_1 = torch.abs(canvas1-gt)
    l1_1[l1_1 > clip] = clip

    reward = (l1_0 * mask).mean(1).mean(1).mean(1) - (l1_1 * mask).mean(1).mean(1).mean(1)
    return reward, mask

#content + style
content_layers = ['r43']
style_layers = ['r12','r22','r33','r43', 'r53']
style_weight = 0.5
content_weight = 1e0
style_scale = 1e10
content_scale = 1e2
loss_layers = style_layers + content_layers
loss_fns =[cal_style_loss] * len(style_layers) + [cal_content_loss] * len(content_layers)
#style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
style_weights = [6, 2, 1, 0.8, 0.2]
content_weights = [1e0]
weights = style_weights + content_weights

# # one style image
# content_initial = None
# style_initial = None
# style_img_ = cv2.imread('./van_gogh.jpg', cv2.IMREAD_UNCHANGED)
# width = 128
# style_img_ = cv2.resize(style_img_, (width, width))
# style_img = torch.tensor(style_img_).float().to(device)
# style_img = style_img[None,:]
# style_img = style_img.permute(0,3,1,2)
# # print(style_img.shape)
# style_targets_img = [GramMatrix()(A).detach() for A in vgg(style_img, style_layers)]


class DDPG(object):
    def __init__(self, batch_size=64, env_batch=1, max_step=40, \
                 tau=0.001, discount=0.9, rmsize=800, \
                 writer=None, resume=None, output_path=None, loss_mode = None, style_type = None):

        self.max_step = max_step
        self.env_batch = env_batch
        self.batch_size = batch_size
        self.state_size = 9
        self.add = 3
        if loss_mode == 'cml1' or loss_mode == 'cml1+style':
            if style_type == 'img':
                self.state_size = 13
                self.add = 5
            else:
                self.state_size = 10
                self.add = 5
        self.actor = ResNet(self.state_size, 18, 65) # target, canvas, stepnum, coordconv 3 + 3 + 1 + 2
        self.actor_target = ResNet(self.state_size, 18, 65)
        self.critic = ResNet_wobn(self.add + self.state_size, 18, 1) # add the last canvas for better prediction
        self.critic_target = ResNet_wobn(self.add + self.state_size, 18, 1)


        self.actor_optim  = Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim  = Adam(self.critic.parameters(), lr=1e-2)
        self.loss_mode = loss_mode
        self.style_type = style_type
        if (resume != None):
            self.load_weights(resume)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = rpm(rmsize * max_step)

        # Hyper-parameters
        self.tau = tau
        self.discount = discount

        # Tensorboard
        self.writer = writer
        self.log = 0

        self.state = [None] * self.env_batch # Most recent state
        self.action = [None] * self.env_batch # Most recent action
        self.choose_device()

    def play(self, state, target=False):
        if self.loss_mode == 'cml1' or self.loss_mode == 'cml1+style':
            if self.style_type == 'img':
              state = torch.cat((state[:, :9].float() / 255,  #canvas and target \
                               state[:, 9:10].float() / 255, # mask \
                               state[:, 9+1:10+1].float() / self.max_step, # step num \
                               coord.expand(state.shape[0], 2, 128, 128)), 1)
            else:
              state = torch.cat((state[:, :6].float() / 255,  #canvas and target \
                               state[:, 6:7].float() / 255, # mask \
                               state[:, 6+1:7+1].float() / self.max_step, # step num \
                               coord.expand(state.shape[0], 2, 128, 128)), 1)
        else:
            state = torch.cat((state[:, :6].float() / 255, state[:, 6:7].float() / self.max_step, coord.expand(state.shape[0], 2, 128, 128)), 1)
        if target:
            return self.actor_target(state)
        else:
            return self.actor(state)

    def update_gan(self, state):
        canvas = state[:, :3]
        gt = state[:, 3 : 6]
        fake, real, penal = update(canvas.float() / 255, gt.float() / 255)
        if self.log % 20 == 0:
            self.writer.add_scalar('train/gan_fake', fake, self.log)
            self.writer.add_scalar('train/gan_real', real, self.log)
            self.writer.add_scalar('train/gan_penal', penal, self.log)

    def evaluate(self, state, action, target=False, mask = None):
        T = state[:, 6 : 7]
        gt = state[:, 3 : 6].float() / 255
        canvas0 = state[:, :3].float() / 255
        canvas1 = decode(action, canvas0)
        if self.loss_mode == 'cm' or self.loss_mode == 'cml1+style':
            T = state[:, 6+1 : 7+1]
            mask = state[:, 6:7].float() / 255

        reward = 0
        if self.loss_mode == 'gan':
          reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)
        elif self.loss_mode == 'l2':
          reward = ((canvas0 - gt) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1)
        elif self.loss_mode == 'cml1':
          reward, mask = content_mask_l1_reward(canvas0, canvas1, gt)
        elif self.loss_mode == 'style':
          reward = cal_perceptual_style_reward(canvas0, canvas1, gt)
        elif self.loss_mode == 'cml1+style':
            if self.style_type == 'img':
                reward, mask = cml1_style_reward_style_img(canvas0, canvas1, gt)
            else:
                reward, mask = cml1_style_reward_style_dataset(canvas0, canvas1, gt)
        coord_ = coord.expand(state.shape[0], 2, 128, 128)
        if self.loss_mode == 'cml1' or self.loss_mode == 'cml1+style':
            if self.style_type == 'img':
                style_img_ref = style_img.repeat(canvas0.shape[0], 1, 1, 1)
                merged_state = torch.cat([canvas0, canvas1, gt, style_img_ref, mask, (T + 1).float() / self.max_step, coord_], 1)
            else:
                merged_state = torch.cat([canvas0, canvas1, gt, mask, (T + 1).float() / self.max_step, coord_], 1)
        else:
          merged_state = torch.cat([canvas0, canvas1, gt, (T + 1).float() / self.max_step, coord_], 1)

        # canvas0 is not necessarily added
        #print(merged_state.shape)
        if target:
            Q = self.critic_target(merged_state)
            return (Q + reward), reward
        else:
            Q = self.critic(merged_state)
            if self.log % 20 == 0:
                self.writer.add_scalar('train/expect_reward', Q.mean(), self.log)
                self.writer.add_scalar('train/reward', reward.mean(), self.log)
            return (Q + reward), reward

    def update_policy(self, lr):
        self.log += 1

        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]

        # Sample batch
        state, action, reward, \
            next_state, terminal, mask = self.memory.sample_batch(self.batch_size, device)
        if self.loss_mode == 'gan':
          self.update_gan(next_state)

        with torch.no_grad():
            next_action = self.play(next_state, True)
            target_q, _ = self.evaluate(next_state, next_action, True, mask)
            target_q = self.discount * ((1 - terminal.float()).view(-1, 1)) * target_q

        cur_q, step_reward = self.evaluate(state, action, mask = mask)
        target_q += step_reward.detach()

        value_loss = criterion(cur_q, target_q)
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        action = self.play(state)
        pre_q, _ = self.evaluate(state.detach(), action, mask = mask)
        policy_loss = -pre_q.mean()
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return -policy_loss, value_loss

    def observe(self, reward, state, done, step, mask):
        s0 = torch.tensor(self.state, device='cpu')
        a = to_tensor(self.action, "cpu")
        r = to_tensor(reward, "cpu")
        s1 = torch.tensor(state, device='cpu')
        d = to_tensor(done.astype('float32'), "cpu")
        m = mask.cpu().clone().detach() if mask is not None else None
        for i in range(self.env_batch):
          if mask is not None:
            mask_add = m[i]
          else:
            mask_add = None
          self.memory.append([s0[i], a[i], r[i], s1[i], d[i], mask_add])
        self.state = state

    def noise_action(self, noise_factor, state, action):
        noise = np.zeros(action.shape)
        for i in range(self.env_batch):
            action[i] = action[i] + np.random.normal(0, self.noise_level[i], action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)

    def select_action(self, state, return_fix=False, noise_factor=0):
        self.eval()
        with torch.no_grad():
            action = self.play(state)
            action = to_numpy(action)
        if noise_factor > 0:
            action = self.noise_action(noise_factor, state, action)
        self.train()
        self.action = action
        if return_fix:
            return action
        return self.action

    def reset(self, obs, factor):
        self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    def load_weights(self, path):
        if path is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))
        if self.loss_mode == 'gan':
          load_gan(path)

    def save_model(self, path):
        self.actor.cpu()
        self.critic.cpu()
        torch.save(self.actor.state_dict(),'{}/actor.pkl'.format(path))
        torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(path))
        save_gan(path)
        self.choose_device()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def choose_device(self):
        Decoder.to(device)
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)