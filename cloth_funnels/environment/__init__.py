from .simEnv import SimEnv, RLEnv
from gymnasium.envs.registration import register


__all__ = [
    'SimEnv',
    'RLEnv']

register(
    id='Cloth-v0',
    entry_point=RLEnv,
    max_episode_steps=50,
)