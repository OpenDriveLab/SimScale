TRAIN_TEST_SPLIT=YOUR_DATASET # e.g. navtrain, synthetic_reaction_pdm_v1.0-0 ...
CACHE_PATH=${NAVSIM_EXP_ROOT}/cache/${TRAIN_TEST_SPLIT}_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=gtrs_dense_vov \
experiment_name=cache_dataset_$TRAIN_TEST_SPLIT \
cache_path=$CACHE_PATH \
worker.threads_per_node=32
