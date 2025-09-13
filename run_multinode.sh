export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
# Change for multinode config
export MASTER_ADDR=$ARNOLD_WORKER_0_HOST
export MASTER_PORT=6004
export NNODES=$ARNOLD_NUM
export GPUS_PER_NODE=$ARNOLD_WORKER_GPU
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export NODE_RANK=$ARNOLD_ID

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  main_fsdp.py --model_name meta-llama/Llama-2-70b-hf --sequence_length 1024
