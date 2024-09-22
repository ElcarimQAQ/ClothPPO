export CUDA_VISIBLE_DEVICES=0
. ./prepare.sh

# 只训练critc不需要test
python cloth_funnels/run_ppo.py\
    name="ppo-only-critic-activation-tanh-lr-1e-4-no-active-v" \
    load=./models/10-05-1335-latest_ckpt.pth \
    num_processes=1 \
    episode_length=10 \
    wandb=online \
    fold_finish=False \
    dump_visualizations=False \
    ray_local_mode=True \
    batch_size=1 \
    warmup=1 \
    gui=False \
    lr=1e-4 \
    test_in_train=False \
    episode_per_test=5 \
    action_primitives='[place]' \
    only_critic=True \
    max_epoch=50 \
    seed=42 \
    repeat_per_collect=5
