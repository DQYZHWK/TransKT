# wo.semantic_embedding
# CUDA_VISIBLE_DEVICES=6 python -u main.py --lr 0.0001 --batch_size 128 --model_id ours_woSE --num_epoch 200  --optim adamw  --IM 1 --theta1 0.15 --theta2 0.6  \
#   --lambda 0.6 --eta 0.7  --warm_epoch 0 --semantic 0 --GNNL 2 > logs/ours_woSE.log 2>&1&

# load pre_trained semantic_embedding
CUDA_VISIBLE_DEVICES=0 python -u main.py --lr 0.0001 --batch_size 128 --model_id ours --num_epoch 200  --optim adamw  --IM 1 --theta1 0.15 --theta2 0.6  \
  --lambda 0.6 --eta 0.7  --warm_epoch 0 --semantic 1 --GNNL 2 > logs/ours.log