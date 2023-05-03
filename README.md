# PaintRL2023

This repo contains project codes for "Exploring Human-like Artistic Stylization through Learning to Paint". This project explores the human-like artisitc stylization with a model-based deep reinforcement learning painting agent. The code is adapted from the [repo](https://github.com/megvii-research/ICCV2019-LearningToPaint) by [Huang et al. 2019](https://arxiv.org/abs/1903.04411).

## Train the model

Monitor the training progress using: `$ tensorboard --logdir=train_log --port=6006`

### Train Neural Renderer

```
$ python3 train_renderer.py
```

### Train the Actor

Please follow the code chunks in `Paint.ipynb` and download the training data. For instance, train with Approach 1 and Modified Perceptual Loss (CM+L1 & Style Loss), then run:

```
$ python3 train.py --debug --batch_size=8 --max_step=80 --loss_mode=cml1+style --dataset=celeb --canvas_color=white --style_type=dataset
```
