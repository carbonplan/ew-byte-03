#!/bin/bash
#
#SBATCH --job-name=run_erw
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --output=run_erw_%a.out
#SBATCH --array=0-63%20

# Preamble (get start time, print SLURM env variables, and exit on error)
start=$(date +%s)
print_job_info
set -euo pipefail

# Activate MIN3P environment
spack env activate /home/AD.UNLV.EDU/perzan/software/spack/env/min3p

# List of sites, forcing types and scenarios
all_sites=("Cecil" "Flanagan" "HoustonBlack" "Kalamazoo" "Kuma" "Palouse" "Pullman" "Yolo")
forcing_types=("longterm" "monthly" "daily" "hourly")
scenarios=("ctrl" "erw")

# Get the site, scenario and forcing type from the SLURM_ARRAY_TASK_ID
n_forcing=${#forcing_types[@]}
n_scenarios=${#scenarios[@]}
site_idx=$(( SLURM_ARRAY_TASK_ID / (n_forcing * n_scenarios) ))
forcing_idx=$(( (SLURM_ARRAY_TASK_ID / n_scenarios) % n_forcing ))
scenario_idx=$(( SLURM_ARRAY_TASK_ID % n_scenarios ))
site=${all_sites[$site_idx]}
forcing_type=${forcing_types[$forcing_idx]}
scenario=${scenarios[$scenario_idx]}
echo "Task ID $SLURM_ARRAY_TASK_ID: $site $forcing_type $scenario"

# Set simulation and database directory
SIM_DIR=$(printf "%s/%s_%s" $site $forcing_type $scenario)
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

echo "Running $SIM_DIR...."

# Run MIN3P
run_min3p

# Collect the results and clean up
cd "$SLURM_SUBMIT_DIR"
collect_min3p "$run_dir"

echo "Finished running spin-up for $site."
print_runtime $start $(date +%s)
echo
