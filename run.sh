export LD_LIBRARY_PATH=/opt/tiger/vescale/nccl/ext-tuner/basic:$LD_LIBRARY_PATH
export NCCL_TUNER_PLUGIN=basic
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV

torchrun --nproc_per_node=8 main_fsdp.py


# export NCCL_TUNER_PLUGIN=/opt/tiger/vescale/nccl/ext-tuner/basic/libnccl-tuner-basic.so