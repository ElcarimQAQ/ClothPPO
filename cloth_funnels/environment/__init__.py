from .cloth_action_gym import SimEnv, RLEnv
from gymnasium.envs.registration import register, registry

__all__ = [
    'SimEnv',
    'RLEnv']


env_id = 'ClothActionGym-v0'
if env_id not in registry:
    register(
        id=env_id,
        entry_point='cloth_funnels.environment.ClothActionGym:RLEnv',
        max_episode_steps=50,
    )