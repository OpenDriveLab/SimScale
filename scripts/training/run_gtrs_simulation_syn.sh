export split=YOUR_SYN_DATASET # e.g. navtrain_reaction_pdm_v1.0-0
export part=1 # 1,2, ..., 32
export PROGRESS_MODE=gen_gt
export POSTFIX=v3

# threads_per_node should be tuned according to the machine's memory limit, if it is too large, ray_distributed will crash.
python $NAVSIM_DEVKIT_ROOT/navsim/agents/tools/gen_vocab_score.py \
train_test_split=${split}_${part} \
experiment_name=debug \
worker.threads_per_node=64 \
+save_name=${split}_${part} \
metric_cache_path=${NAVSIM_EXP_ROOT}/cache/${split}_metric_cache \
scorer.config.human_penalty_filter=False 