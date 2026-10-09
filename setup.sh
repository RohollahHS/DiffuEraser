git clone https://github.com/RohollahHS/DiffuEraser
cd DiffuEraser

module purge all --force

module load python/3.10
virtualenv /scratch/rohhs/venvs/diffueraser
source /scratch/rohhs/venvs/diffueraser/bin/activate
module load StdEnv/2023  nvhpc/23.9  openmpi/4.1.5
module load cuda/12.2

export PIP_CONFIG_FILE=''
export PYTHONPATH=''

pip install pip --upgrade

pip install -r requirements.txt

# ln -s $HF_HUB/ weights