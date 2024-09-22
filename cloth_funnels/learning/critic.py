import logging
from typing import (
    Sequence,
    Union,
)

from.nets import MaximumValuePolicy
from time import time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MaximumValueCritic(MaximumValuePolicy):
    def __init__(self, action_shape: Union[int, Sequence[int]], last_size: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = int(np.prod(action_shape))
        self.last = layer_init(nn.Linear(self.input_dim, last_size), std=1.0)
        # Trick10: use tanh
        self.activation_fn = nn.Tanh()

    def forward(self, obs, state=None, info={}):
        # 输入batch 转为list
        obs_list = []
        for i in range(obs.shape[0]):
            obs_list.append(obs[i])

        outs = self.act(obs_list, state)

        # 结果转为batch
        vmaps = torch.stack(outs, dim=0)
        #  TODO: 激活logits 实验： 11-27-2210-ppo-only-critic,  11-27-2144-ppo-only-critic
        # logits = vmaps.flatten(1)
        # logits = self.last(logits)
        # v = self.activation_fn(logits) * 2 + 1
        logits = vmaps.flatten(1)
        v = self.last(logits)
        logging.debug(f"critic logits sum: {logits.sum()} ,v : {v}")
        return v

    def get_action_single(self, obs, explore=True):
        state = dict(obs)
        obs = obs['transformed_obs']
        for primitive in self.action_primitives:
            value_maps = self.value_net.forward_for_optimize(obs, primitive, preprocess=True)
            mask = state[f'{primitive}_mask']
            vmap = (1 - self.deformable_weight) * value_maps['rigid'][primitive].squeeze(1) + \
                   self.deformable_weight * value_maps['deformable'][primitive].squeeze(1)
            vmap = self.activation_fn(vmap)
        return vmap


    def act(self, obs, state):
        start = time()
        logging.debug("Starting policy.act()")
        r = [self.get_action_single(o, explore=False) for o in obs]
        end = time()
        logging.debug("[Critic] Forward took: ", end - start, "with #obs: ", len(obs))
        return r