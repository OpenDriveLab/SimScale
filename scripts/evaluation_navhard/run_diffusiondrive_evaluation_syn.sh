split=navhard
agent=diffusiondrive_agent
metric_cache_path="${NAVSIM_EXP_ROOT}/cache/${split}_two_stage_metric_cache"

#--------------------------
export SYN_IDX=0 # 0, 1, 2, 3, 4
export SYN_GT=pdm # pdm, recovery
#--------------------------

postfix=v1.0-${SYN_IDX}

dir=train_diffusiondrive_resnet_syn_react_${SYN_GT}_${postfix}

echo "Starting evaluation script..."
for epoch in $(seq 0 99); do
    padded_epoch=$(printf "%02d" $epoch)
    experiment_name="${dir}/test-${padded_epoch}ep-${split}"
    ckpt="'${NAVSIM_EXP_ROOT}/${dir}/epoch=${padded_epoch}.ckpt'"

    python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2_diffusiondrive.py \
        agent=$agent \
        dataloader.params.batch_size=32 \
        agent.checkpoint_path=${ckpt} \
        experiment_name=${experiment_name} \
        +cache_path=null \
        metric_cache_path=${metric_cache_path} \
        train_test_split=${split}_two_stage
done
