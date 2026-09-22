#!/bin/bash  
#  
#SBATCH --job-name=met_transport_peak_tracer_time
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1  
#SBATCH --nodes=1
#SBATCH --output=met_transport_peak_tracer_time.out

# Preamble (get start time, print SLURM env variables, and exit on error)
start=$(date +%s)
print_job_info
set -e

source /home/AD.UNLV.EDU/perzan/software/python/venv/min3p-env/.venv/bin/activate

while read START_DATE
do
    # Note that 2007-02-27 never finished running
    if [[ "$START_DATE" == "2007-02-27" ]]; then
        echo "(skipping $START_DATE)"
        continue
    fi

    OUTFILE="output_data/${START_DATE}_TracerBreakthrough.csv"
    echo "Running $START_DATE"
    python -m byte_util.cli.calculate_peak_tracer_times $START_DATE \
              --outfile "$OUTFILE" \
              --scenarios hourly daily monthly

done < start_dates.txt

echo "Finished calculating peak tracer times $(wc -l start_dates.txt)"

print_runtime $start $(date +%s) 
