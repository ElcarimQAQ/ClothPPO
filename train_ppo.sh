export CUDA_VISIBLE_DEVICES=2
. ./prepare.sh
python cloth_funnels/run_ppo.py\
    name="pretrained-mlp1-length-rewardsaclemax5-0.9-unravel-place" \
    load=./models/longsleeve_canonicalized_alignment.pth \
    num_processes=1 \
    episode_length=10 \
    wandb=online \
    fold_finish=False \
    dump_visualizations=False \
    ray_local_mode=False \
    batch_size=1 \
    warmup=1 \
    gui=False \
    lr=1e-4 \
    test_in_train=True \
    episode_per_test=5 \
    action_primitives='[place]' \
    max_epoch=100 \
    seed=42 \
    repeat_per_collect=5 \
    log_dir="./all_experiments/" \
    cont=None \
    resume=False \
    test_processes=6 \
    reward_normalization=True \
    test_dataset_size=45 \
    reward_type="right_length_reward:-1" \
    tags='[pretrain]'


