export CUDA_VISIBLE_DEVICES=0,2
export OMP_NUM_THREADS=2

python -m torch.distributed.launch --nproc_per_node=2 src/train_model.py
