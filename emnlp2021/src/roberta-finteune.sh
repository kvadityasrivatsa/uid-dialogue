#!/bin/bash

#SBATCH -p long    # which partition to run on ('compute' is default)
#SBATCH -J entropy-empathy    # arbitrary name for the job (you choose)
#SBATCH --output=roberta-pbft-entropy-%j.out
#SBATCH --error=roberta-pbft-entropy-%j.err
#SBATCH --cpus-per-task=4    # tell Slurm how many CPU cores you need, if different from default; your job won't be able to use more than this
#SBATCH --mem=50000    # how much RAM you need (30GB in this case), if different from default; your job won't be able to use more than this
#SBATCH -t 1-02:30:00    # maximum execution time: in this case one day, two hours and thirty minutes (optional)



# Uncomment the following to get a log of memory usage; NOTE don't use this if you plan to run multiple processes in your job and you are placing "wait" at the end of the job file, else Slurm won't be able to tell when your job is completed!

if [ "$#" -ne 1 ]; then
    echo "Usage: bash run_roberta_ft.sh <dataset_path>"
    exit 1
fi

DATASET_PATH=$1

DATASET_NAME=$(basename "$DATASET_PATH" .csv)

vmstat -S M 5 >> memory_usage_$SLURM_JOBID.log &

module load anaconda3-2024.02-1
source /home/adebnath/anaconda3/bin/activate
eval "${conda shell.bash hook}"

# Your commands here
conda activate base
python3 roberta_ft.py "DATASET_PATH"
conda deactivate
