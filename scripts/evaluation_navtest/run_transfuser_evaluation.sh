split=navtest
agent=transfuser_agent
metric_cache_path="${NAVSIM_EXP_ROOT}/cache/${split}_metric_cache"

dir=train_transfuser_resnet

echo "Starting evaluation script..."
for epoch in $(seq 0 99); do
    padded_epoch=$(printf "%02d" $epoch)

    experiment_name="${dir}/test-${padded_epoch}ep-${split}"
    ckpt="'${NAVSIM_EXP_ROOT}/${dir}/epoch=${padded_epoch}.ckpt'"

    python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage_gpu_transfuser.py \
        agent=$agent \
        dataloader.params.batch_size=32 \
        agent.checkpoint_path=${ckpt} \
        experiment_name=${experiment_name} \
        +cache_path=null \
        metric_cache_path=${metric_cache_path} \
        train_test_split=${split}
done