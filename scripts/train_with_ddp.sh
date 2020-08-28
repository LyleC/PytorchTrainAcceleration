export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=2

python -m torch.distributed.launch --nproc_per_node=4 src/train_model.py
