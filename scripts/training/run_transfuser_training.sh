config="default_training" # this config uses navtrain train logs for training; val logs for validation
agent=transfuser_agent

lr=1e-4
bs=64 
max_epochs=100

experiment_name=train_transfuser_resnet

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_transfuser.py \
        --config-name ${config} \
        agent=$agent \
        experiment_name=$experiment_name \
        train_test_split=navtrain \
        trainer.params.max_epochs=$max_epochs \
        +agent.config.ckpt_path=${experiment_name} \
        agent.lr=${lr} \
        cache_path=${NAVSIM_EXP_ROOT}/cache/trainval_cache \
        use_cache_without_dataset=True  \
        force_cache_computation=False \
        +resume_ckpt_path=${experiment_name}/last.ckpt 
