CUDA_VISIBLE_DEVICES=0 python cloth_funnels/run_sim.py \
name="demo-single" \
load=./models/longsleeve_canonicalized_alignment.pth \
eval_tasks=./assets/tasks/longsleeve-single.hdf5 \
eval=True \
num_processes=1 \
episode_length=10 \
wandb=disabled \
fold_finish=True \
dump_visualizations=True