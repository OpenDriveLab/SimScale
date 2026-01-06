config="default_training" # this config uses navtrain train logs for training; val logs for validation
agent=diffusiondrive_agent

# main config
#--------------------------
export SYN_IDX=0 # 0, 1, 2, 3, 4
export SYN_GT=recovery # pdm, recovery
#--------------------------

postfix=v1.0-${SYN_IDX}

lr=6e-4
bs=64
max_epochs=100

experiment_name=train_diffusiondrive_resnet_syn_react_${SYN_GT}_${postfix}

echo "Starting training script..."
torchrun --nproc_per_node=8 \
        $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_diffusiondrive.py \
        --config-name ${config} \
        agent=$agent \
        experiment_name=$experiment_name \
        train_test_split=navtrain \
        dataloader.params.batch_size=${bs} \
        trainer.params.max_epochs=$max_epochs \
        +agent.config.ckpt_path=${experiment_name} \
        agent.lr=${lr} \
        cache_path=${NAVSIM_EXP_ROOT}/cache/trainval_cache \
        use_cache_without_dataset=True  \
        force_cache_computation=False \
        +resume_ckpt_path=${experiment_name}/last.ckpt 