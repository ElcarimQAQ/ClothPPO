export CUDA_VISIBLE_DEVICES=2
. ./prepare.sh

EXP_NAME=clothfunnles-place
TASK="multi-dress-eval"
NAME="${TASK}${EXP_NAME}"

python cloth_funnels/run_sim.py\
    name="${NAME}" \
    load=./models/longsleeve_canonicalized_alignment.pth \
    eval_tasks="./assets/tasks/${TASK}.hdf5" \
    eval=True \
    num_processes=2 \
    episode_length=10 \
    wandb=disabled \
    fold_finish=False \
    dump_visualizations=True 

python cloth_funnels/environment/visualize.py ./experiments/*${NAME}/replay_buffer.hdf5 n
