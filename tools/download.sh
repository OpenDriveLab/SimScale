cd $HOME # Change to your desired download directory

repo="https://huggingface.co/datasets/OpenDriveLab/SimScale/resolve/main"

# 1. simulation data with planner-based pseudo-expert
rounds=5
splits=(66,56,47,39,33)

for round in $(seq 0 $((rounds - 1))); do
    # 1. meta data
    wget ${repo}/SimScale_data/synthetic_reaction_pdm_v1.0-${round}/simscale_pdm_v1.0-${round}_meta_datas.tar.gz
    echo "[Planner-based] Downloaded meta data for round ${round}"
    tar -xzvf simscale_pdm_v1.0-${round}_meta_datas.tar.gz
    rm simscale_pdm_v1.0-${round}_meta_datas.tar.gz

    # 2. hist sensor data
    split=${splits[$round]}
    for idx in $(seq 0 $((split - 1))); do
        wget ${repo}/SimScale_data/synthetic_reaction_pdm_v1.0-${round}/simscale_pdm_v1.0-${round}_sensor_blobs_hist/simscale_pdm_v1.0-${round}_sensor_blobs_hist_${idx}.tar.gz
        echo "[Planner-based] Downloaded hist sensor data ${idx} for round ${round}"
        tar -xzvf simscale_pdm_v1.0-${round}_sensor_blobs_hist_${idx}.tar.gz
        rm simscale_pdm_v1.0-${round}_sensor_blobs_hist_${idx}.tar.gz
    done

    # 3. future sensor data
    # [TBD]

done

# 2. simulation data with recovery-based pseudo-expert
rounds=5
splits=(45,36,28,22,17)

for round in $(seq 0 $((rounds - 1))); do
    # 1. meta data
    wget ${repo}/SimScale_data/synthetic_reaction_recovery_v1.0-${round}/simscale_recovery_v1.0-${round}_meta_datas.tar.gz
    echo "[Recovery-based] Downloaded meta data for round ${round}"
    tar -xzvf simscale_recovery_v1.0-${round}_meta_datas.tar.gz
    rm simscale_recovery_v1.0-${round}_meta_datas.tar.gz

    # 2. hist sensor data
    split=${splits[$round]}
    for idx in $(seq 0 $((split - 1))); do
        wget ${repo}/SimScale_data/synthetic_reaction_recovery_v1.0-${round}/simscale_recovery_v1.0-${round}_sensor_blobs_hist/simscale_recovery_v1.0-${round}_sensor_blobs_hist_${idx}.tar.gz
        echo "[Recovery-based] Downloaded hist sensor data ${idx} for round ${round}"
        tar -xzvf simscale_recovery_v1.0-${round}_sensor_blobs_hist_${idx}.tar.gz
        rm simscale_recovery_v1.0-${round}_sensor_blobs_hist_${idx}.tar.gz
    done

    # 3. future sensor data
    # [TBD]

done