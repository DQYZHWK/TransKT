# im
CUDA_VISIBLE_DEVICES=3 python -u main.py --lr 0.0001 --batch_size 64 --model_id ours --num_epoch 200  --optim adamw  --IM 1 --theta1 0.2 --theta2 0.7  \
  --lambda 0.8 --eta 0.4  --warm_epoch 0 --GNNL 2 > log/test.log 2>&1&