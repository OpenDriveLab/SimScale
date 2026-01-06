# Download and installation


### 1. Clone the devkit

Clone the repository

```bash
git clone https://github.com/NVlabs/GTRS.git
cd GTRS
```

### 2. Download the dataset

You need to download the OpenScene logs and sensor blobs, as well as the nuPlan maps.
We provide scripts to download the nuplan maps, the mini split and the test split.
Navigate to the download directory and download the maps

**NOTE: Please check the [LICENSE file](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) before downloading the data.**

```bash
cd download && ./download_maps
```

Next download the data splits you want to use.
Note that the dataset splits do not exactly map to the recommended standardized training / test splits-
Please refer to [splits](splits.md) for an overview on the standardized training and test splits including their size and check which dataset splits you need to download in order to be able to run them.
You can download these splits with the following scripts.

```bash
./download_mini
./download_trainval
./download_test
./download_warmup_two_stage
./download_navhard_two_stage
./download_private_test_hard_two_stage
```

Also, the script `./download_navtrain` can be used to download a small portion of the  `trainval` dataset split which is needed for the `navtrain` training split.

Execute the following script to download the simulated ground-truths for different vocabularies:
```bash
cd ~/navsim_workspace/dataset;
mkdir traj_pdm_v2; cd traj_pdm_v2;
# ground-truths without data augmentations
mkdir ori; cd ori;
wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/navtrain_8192.pkl
wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/navtrain_16384.pkl
# ground-truths with data augmentations
cd ../; mkdir random_aug; cd random_aug;
wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/rot_30-trans_0-va_0-p_0.5-ensemble.json
wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/aug_traj_pdm.zip
unzip aug_traj_pdm.zip
rm aug_traj_pdm.zip
```

Execute the following script to download the pretrained vision backbones:
```bash
cd ~/navsim_workspace/dataset;
mkdir models; cd models;
wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/dd3d_det_final.pth
```

The final folder should have the following structure:
```angular2html
~/navsim_workspace
├── GTRS (containing the devkit)
├── exp
└── dataset
    ├── maps
    ├── models
    |    └── dd3d_det_final.pth
    ├── traj_pdm_v2
    |    ├── ori
    |    |   ├── navtrain_16384.pkl
    |    |   └── navtrain_8192.pkl
    |    ├── random_aug
    |    |   ├── rot_30-trans_0-va_0-p_0.5-ensemble.json
    |    |   └── rot_30-trans_0-va_0-p_0.5-ensemble
    |    |       └── split_pickles
    |    |           ├──  xxx.pkl
    |    |           ├──  ...
    ├── navsim_logs
    |    ├── test
    |    ├── trainval
    |    ├── private_test_hard
    |    |         └── private_test_hard.pkl
    │    └── mini
    └── sensor_blobs
    |    ├── test
    |    ├── trainval
    |    ├── private_test_hard
    |    |         ├──  CAM_B0
    |    |         ├──  CAM_F0
    |    |         ├──   ...
    |    └── mini
    └── navhard_two_stage
    |    ├── openscene_meta_datas
    |    ├── sensor_blobs
    |    ├── synthetic_scene_pickles
    |    └── synthetic_scenes_attributes.csv
    └── warmup_two_stage
    |    ├── openscene_meta_datas
    |    ├── sensor_blobs
    |    ├── synthetic_scene_pickles
    |    └── synthetic_scenes_attributes.csv
    └── private_test_hard_two_stage
         ├── openscene_meta_datas
         └── sensor_blobs

```
Set the required environment variables, by adding the following to your `~/.bashrc` file
Based on the structure above, the environment variables need to be defined as:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/GTRS"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
export NAVSIM_TRAJPDM_ROOT="$HOME/navsim_workspace/dataset/traj_pdm_v2"
```

⏰ **Note:** The `navhard_two_stage` split is used for local testing of your model's performance in a two-stage pseudo closed-loop setup.
In contrast, `warmup_two_stage` is a smaller dataset designed for validating and testing submissions to the [Hugging Face Warmup leaderboard](https://huggingface.co/spaces/AGC2025/e2e-driving-warmup).
In other words, the results you obtain locally on `warmup_two_stage` should match the results you see after submitting to Hugging Face.
`private_test_hard_two_stage` contains the challenge data.
You will need it to generate a `submission.pkl` in order to participate in the official challenge on the [Hugging Face CPVR 2025 leaderboard](https://huggingface.co/spaces/AGC2025/e2e-driving-internal) (for more details, see [Submission](submission.md)).


### 3. Install the navsim-devkit

Finally, install navsim.
To this end, create a new environment and install the required dependencies:

```bash
conda env create --name conda_gtrs -f environment.yml
conda activate conda_gtrs
pip install --upgrade diffusers[torch]
pip install -e .
```