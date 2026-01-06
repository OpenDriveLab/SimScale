cd $HOME # Change to your desired download directory
OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"

# 1. remove simulation data with planner-based pseudo-expert
rounds=5
for round in $(seq 0 $((rounds - 1))); do
    name="synthetic_reaction_pdm_v1.0-${round}"
    # meta data
    mv -f SimScale/${name}/meta_datas ${OPENSCENE_DATA_ROOT}/navsim_logs/${name}
    # sensor data
    mv -f SimScale/${name}/sensor_blobs_hist ${OPENSCENE_DATA_ROOT}/sensor_blobs/${name}
    # [TBD] mv -f SimScale/${name}/sensor_blobs_fut ${OPENSCENE_DATA_ROOT}/sensor_blobs/${name}
done

# 2. remove simulation data with recovery-based pseudo-expert
rounds=5
for round in $(seq 0 $((rounds - 1))); do
    name="synthetic_reaction_recovery_v1.0-${round}"
    # meta data
    mv -f SimScale/${name}/meta_datas ${OPENSCENE_DATA_ROOT}/navsim_logs/${name}
    # sensor data
    mv -f SimScale/${name}/sensor_blobs_hist ${OPENSCENE_DATA_ROOT}/sensor_blobs/${name}
    # [TBD] mv -f SimScale/${name}/sensor_blobs_fut ${OPENSCENE_DATA_ROOT}/sensor_blobs/${name}
done