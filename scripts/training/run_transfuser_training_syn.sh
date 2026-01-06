config="default_training"  # this config uses navtrain train logs for training; val logs for validation
agent=transfuser_agent

# main config
#--------------------------
export SYN_IDX=0 # 0, 1, 2, 3, 4
export SYN_GT=pdm # pdm, recovery
#--------------------------

postfix=v1.0-${SYN_IDX}

lr=2.82e-4
bs=64
max_epochs=100

experiment_name=train_transfuser_resnet_syn_react_${SYN_GT}_${postfix}

echo "Starting training script..."
python  ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_transfuser.py \
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