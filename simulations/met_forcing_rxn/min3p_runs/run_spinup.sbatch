#!/bin/bash
#
#SBATCH --job-name=run_spinup
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --output=run_spinup_%a.out
#SBATCH --array=0-7%5

# Preamble (get start time, print SLURM env variables, and exit on error)
start=$(date +%s)
print_job_info
set -euo pipefail

# Activate MIN3P environment
spack env activate /home/AD.UNLV.EDU/perzan/software/spack/env/min3p

# Get the site from the SLURM_ARRAY_TASK_ID
all_sites=("Cecil" "Flanagan" "HoustonBlack" "Kalamazoo" "Kuma" "Palouse" "Pullman" "Yolo")
site=${all_sites[$SLURM_ARRAY_TASK_ID]}

SIM_DIR="${site}/spinup"
DATABASE_DIR="${HOME}/min3p_model_bytes/databases"

# Clear old output
cd "$SIM_DIR"
clear_min3poutput
cd "$SLURM_SUBMIT_DIR"

# Save the submission directory and create temporary run directory
run_dir=$(stage_min3p "$SIM_DIR" "$DATABASE_DIR")

cd "$run_dir/run"

# Get the run name from root.dat
prefix=$(cat root.dat)

echo "Running $site...."

# Run MIN3P
run_min3p

# Collect the results and clean up
cd "$SLURM_SUBMIT_DIR"
collect_min3p "$run_dir"

echo "Finished running spin-up for $site."
print_runtime $start $(date +%s)
echo
