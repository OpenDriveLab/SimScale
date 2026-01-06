config="default_training" # this config uses navtrain train logs for training; val logs for validation
agent=gtrs_dense_vov 

# main config
#--------------------------
export SYN_IDX=0 # 0, 1, 2, 3, 4
export SYN_GT=pdm # pdm, recovery
syn_imi=true # true, false
backbone_type=resnet  # vov, resnet
#--------------------------

GPU_NUM=8
NUM_NODES=4
MASTER_ADDR=YOUR_MASTER_ADDR
MASTER_PORT=YOUR_MASTER_PORT

lr=4e-4
bs=11
max_epochs=50

postfix=v1.0-${SYN_IDX}

[ "$syn_imi" = "true" ] && tag="" || tag="_rewards_only"
experiment_name=train_gtrs_dense_${backbone_type}_syn_react_${SYN_GT}_${postfix}${tag}

echo "Starting training script..."
torchrun --nnodes=${NUM_NODES} \
    --nproc_per_node=${GPU_NUM} \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_dense.py \
    --config-name ${config} \
    trainer.params.num_nodes=${NUM_NODES} \
    agent=${agent} \
    experiment_name=${experiment_name} \
    train_test_split=navtrain \
    dataloader.params.batch_size=${bs} \
    trainer.params.max_epochs=${max_epochs} \
    trainer.params.precision=32 \
    agent.pdm_gt_path=${NAVSIM_TRAJPDM_ROOT}/ori/navtrain_16384.pkl \
    agent.config.ckpt_path=${experiment_name} \
    agent.config.backbone_type=${backbone_type} \
    +agent.config.syn_imi=${syn_imi} \
    agent.lr=${lr} \
    cache_path=${NAVSIM_EXP_ROOT}/cache/trainval_cache \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    +resume_ckpt_path=${experiment_name}/last.ckpt \