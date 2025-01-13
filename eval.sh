export CUDA_VISIBLE_DEVICES=0
. ./prepare.sh

EXP_NAME=clothppo
CHECKPOINT_DIR="./models/"
TASK="multi-longsleeve-eval"

CHECKPOINT_PREFIX="ClothPPO"
START_INDEX=42
END_INDEX=42

if [ "$1" = "clothppo" ]; then
  CHECKPOINT_PATH="${CHECKPOINT_DIR}ClothPPO.pth"
  NAME="${TASK}-clothppo"
  python cloth_funnels/run_ppo.py \
          name="${NAME}" \
          load="${CHECKPOINT_PATH}" \
          eval_tasks="./assets/tasks/${TASK}.hdf5" \
          eval=True \
          num_processes=5 \
          episode_length=10 \
          fold_finish=False \
          dump_visualizations=True \
          ray_local_mode=False \
          gui=False \
          resume=False \
          wandb=disabled \
          action_primitives='[place]' \
          test_dataset_size=200

    python cloth_funnels/environment/visualize.py ./*experiments/*${NAME}/replay_buffer.hdf5 n
elif [ "$1" = "best" ]; then
  CHECKPOINT_PATH="${CHECKPOINT_DIR}best_policy.pth"
  NAME="${TASK}-${EXP_NAME}-best"
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
          test_dataset_size=200

    python cloth_funnels/environment/visualize.py ./*experiments/*${NAME}/replay_buffer.hdf5 n
else
  for ((i=$START_INDEX; i>=$END_INDEX; i--)); do
      CHECKPOINT_PATH="${CHECKPOINT_DIR}${CHECKPOINT_PREFIX}${i}.pth"
      NAME="${TASK}-${EXP_NAME}-${CHECKPOINT_PREFIX}${i}"
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
          test_dataset_size=100
      python cloth_funnels/environment/visualize.py ./*experiments/*${NAME}/replay_buffer.hdf5 n
  done
fi