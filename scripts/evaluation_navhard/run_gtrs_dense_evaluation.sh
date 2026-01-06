split=navhard
agent=gtrs_dense_vov
metric_cache_path="${NAVSIM_EXP_ROOT}/cache/${split}_two_stage_metric_cache"

#--------------------------
backbone_type=vov # vov, resnet
#--------------------------

dir=train_gtrs_dense_${backbone_type}

echo "Starting evaluation script..."
for epoch in $(seq 0 49); do
    padded_epoch=$(printf "%02d" $epoch)

    experiment_name="${dir}/test-${padded_epoch}ep-${split}"
    ckpt="'${NAVSIM_EXP_ROOT}/${dir}/epoch=${padded_epoch}.ckpt'"

    export SUBSCORE_PATH=${NAVSIM_EXP_ROOT}/${experiment_name}/epoch${epoch}_${split}.pkl; # save path for the scores

    python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2.py \
        agent=$agent \
        +combined_inference=false \
        dataloader.params.batch_size=16 \
        agent.checkpoint_path=${ckpt} \
        agent.config.vocab_path=${NAVSIM_DEVKIT_ROOT}/traj_final/8192.npy \
        agent.config.backbone_type=${backbone_type} \
        trainer.params.precision=32 \
        experiment_name=${experiment_name} \
        +cache_path=null \
        metric_cache_path=${metric_cache_path} \
        train_test_split=${split}_two_stage
done
