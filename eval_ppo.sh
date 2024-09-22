export CUDA_VISIBLE_DEVICES=0
. ./prepare.sh

EXP_NAME=12-25-0152-ppo-mlp1-rewardsaclemax10-sota-place-tanh-lr1e-4
CHECKPOINT_DIR="./all_experiments/${EXP_NAME}/"
CHECKPOINT_PREFIX="checkpoint_"
START_INDEX=97
END_INDEX=97
TASK="multi-pants-eval"

if [ "$1" = "best" ]; then
  CHECKPOINT_PATH="${CHECKPOINT_DIR}best_policy.pth"
  NAME="${TASK}${EXP_NAME}-best"
  python cloth_funnels/run_ppo.py \
          name="${NAME}" \
          load="${CHECKPOINT_PATH}" \
          eval_tasks="./assets/tasks/${TASK}.hdf5" \
          eval=True \
          num_processes=5 \
          episode_length=10 \
          fold_finish=False \
          dump_visualizations=False \
          ray_local_mode=False \
          gui=False \
          resume=False \
          wandb=disabled \
          action_primitives='[place]' \
          log_dir="./all_experiments/" \

    python cloth_funnels/environment/visualize.py ./*experiments/*${NAME}/replay_buffer.hdf5 n
else
  for ((i=$START_INDEX; i>=$END_INDEX; i--)); do
      CHECKPOINT_PATH="${CHECKPOINT_DIR}${CHECKPOINT_PREFIX}${i}.pth"
      NAME="${TASK}${EXP_NAME}-${CHECKPOINT_PREFIX}${i}"
      python cloth_funnels/run_ppo.py \
          name="${NAME}" \
          load="${CHECKPOINT_PATH}" \
          eval_tasks="./assets/tasks/${TASK}.hdf5" \
          eval=True \
          num_processes=5 \
          episode_length=10 \
          fold_finish=False \
          dump_visualizations=False \
          ray_local_mode=False \
          gui=False \
          resume=False \
          wandb=disabled \
          action_primitives='[place]' \
          log_dir="./all_experiments/" \
          test_dataset_size=250
      python cloth_funnels/environment/visualize.py ./*experiments/*${NAME}/replay_buffer.hdf5 n
  done
fi