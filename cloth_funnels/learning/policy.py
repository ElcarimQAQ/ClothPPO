import logging
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from torch import nn
from torchrl.modules.distributions import MaskedCategorical

from tianshou.policy import PPOPolicy
from tianshou.utils.net.discrete import Actor
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from .utils import unravel_indices

class MaximumValuePPOPolicy(PPOPolicy):
    def __init__(self,
                 actor: torch.nn.Module,
                 critic: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 dist_fn: Type[torch.distributions.Distribution],
                 **kwargs) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any,
    ) -> Batch:
        """  自定义policy """
        if isinstance(self.actor, Actor):
            logits, hidden = self.actor(batch.obs['transformed_obs'], state=state, info=batch.info)
            if isinstance(logits, tuple):
                dist = self.dist_fn(*logits)
            else:
                dist = self.dist_fn(logits)
            if self._deterministic_eval and not self.training:
                if self.action_type == "discrete":
                    act = logits.argmax(-1)
                elif self.action_type == "continuous":
                    act = logits[0]
            else:
                act = dist.sample()
            return Batch(logits=logits, act=act, state=hidden, dist=dist)

        # MaximumValueActor
        else:
            hidden = state
            try:
                logits, max_index, mask = self.actor(batch.obs, state=state, info=batch.info)
                self.actor.env_steps += 1
            except ValueError as e:
                logging.error(f'[MaximumValuePPOPolicy] {e}')

            if self.dist_fn == MaskedCategorical:
                mask_indices = mask.view(mask.shape[0], -1)
                dist = self.dist_fn(logits=logits, mask=mask_indices)
            else:
                dist = self.dist_fn(logits)

            if self._deterministic_eval and not self.training:
                act_indices = logits.argmax(-1)
                act = unravel_indices(act_indices, mask.shape[1:])
                if torch.equal(act.cpu(), torch.Tensor(max_index)) == False:
                    logging.warning(f"[MaximumValuePPOPolicy ] Eval: max_index:{max_index}, argmax act_indices :{act_indices}, act:{act}")
            else:
                try:
                    act_indices = dist.sample()
                except Exception as e:
                    act_indices = logits.argmax(-1)
                    logging.error("[MaximumValuePPOPolicy] dist sample error, get argmax action")
            # max_index = np.unravel_index(act, mask.shape)
            # act = self.actor.restore_action_index(act_indices, mask)
            act = unravel_indices(act_indices, mask.shape[1:])
            hidden = {}
            hidden['act_indices'] = act_indices
            logging.debug("[MaximumValuePPOPolicy] act_indices:{}".format(act_indices))
            return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)

        batch.act = to_torch_as(batch.policy.hidden_state['act_indices'], batch.v_s)

        with torch.no_grad():
            batch.logp_old = self(batch).dist.log_prob(batch.act)
        return batch

    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        """  copy from PPOPolicy.learn() """
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor, only input obs
                dist = self(minibatch).dist
                # TODO: norm 导致了nan, 不norm
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv -
                                     mean) / (std + self._eps)  # per-batch norm
                    logging.info(f"[MaximumValuePPOPolicy] norm_adv is:{minibatch.adv}")
                ratio = (dist.log_prob(minibatch.act) -
                         minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(
                    1.0 - self._eps_clip, 1.0 + self._eps_clip
                ) * minibatch.adv
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # calculate loss for critic.py
                value = self.critic(minibatch.obs).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + \
                             (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self._weight_vf * vf_loss \
                       - self._weight_ent * ent_loss
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                # self.actor.train_steps += 1
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }

