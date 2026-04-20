#!/encs/bin/bash
#SBATCH --job-name=dsan_experiment
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --gres=gpu:nvidia_a100_7g.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=dsan_experiment_%j.out
#SBATCH --error=dsan_experiment_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=x_qiaoyu@speed.encs.concordia.ca

set -euo pipefail

echo "HOST: $(hostname)"
echo "START: $(date)"
nvidia-smi

# Conda
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/qiaoyu/speed-hpc/env/dsan

# Cache (bash version)
source /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh

# Run
cd /speed-scratch/qiaoyu/speed-hpc/project/deep-transfer-learning-DSAN/UDA/pytorch1.0/DSAN
            
python run_pacs_uda_experiments.py \
--root_path /speed-scratch/qiaoyu/speed-hpc/datasets/PACS/pacs_data/pacs_data \
--batch_size 16 \
--nepoch 50 \
--pretrained true \
--gpu 0 \
--weights 0 0.1 0.2 0.3 0.5 \
--seeds 2021 2022 2023 \
--output_dir experiments/pacs_uda

echo "END: $(date)"
