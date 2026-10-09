##### Directories Setup
SRUN_ARGS="--ntasks=$SLURM_NNODES --ntasks-per-node=1"
export XDG_CACHE_HOME=${SLURM_TMPDIR}/.cache
export XDG_CONFIG_HOME=${SLURM_TMPDIR}/.config
export TRITON_CACHE_DIR=${SLURM_TMPDIR}/.cache/triton
export TMPDIR=${SLURM_TMPDIR}/.cache/tmp
export VLLM_CACHE_ROOT=${SLURM_TMPDIR}/vllm_cache
export VLLM_CONFIG_ROOT=${SLURM_TMPDIR}/vllm_config
export VLLM_ASSETS_CACHE=${SLURM_TMPDIR}/assets_cache
export FLASHINFER_WORKSPACE_BASE=${SLURM_TMPDIR}/flashinfer
srun $SRUN_ARGS mkdir -p ${XDG_CACHE_HOME}
srun $SRUN_ARGS mkdir -p ${XDG_CONFIG_HOME}
srun $SRUN_ARGS mkdir -p ${TRITON_CACHE_DIR}
srun $SRUN_ARGS mkdir -p ${TMPDIR}
srun $SRUN_ARGS mkdir -p ${VLLM_CACHE_ROOT}
srun $SRUN_ARGS mkdir -p ${VLLM_CONFIG_ROOT}
srun $SRUN_ARGS mkdir -p ${VLLM_ASSETS_CACHE}
srun $SRUN_ARGS mkdir -p ${FLASHINFER_WORKSPACE_BASE}

ssh $node1 &
proxy --hostname 0.0.0.0 --port 8899 >/dev/null 2>&1 &

export http_proxy=http://${node1}:8899
export https_proxy=$http_proxy
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$http_proxy
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

##### Env Setup
module --force purge all
module load StdEnv/2023  nvhpc/23.9  openmpi/4.1.5
module load cuda/12.2
source /scratch/rohhs/venvs/diffueraser/bin/activate

cd $PROJECTS_DIR/DiffuEraser

##### Run the code

# export HF_HUB_OFFLINE=1

# python run_diffueraser.py

input_video=/scratch/rohhs/downloads/yt-dlp/1009_segments/1009_segments-1.mp4
input_mask=/scratch/rohhs/downloads/yt-dlp/1009_segments/1009_segments-3.mp4
save_path=/scratch/rohhs/downloads/yt-dlp/

# python app.py --save_path $save_path

python run_diffueraser.py --input_video $input_video --input_mask $input_mask --save_path $save_path
