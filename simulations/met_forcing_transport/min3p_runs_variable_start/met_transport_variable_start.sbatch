#!/bin/bash  
#  
#SBATCH --job-name=met_transport_variable_start
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1  
#SBATCH --nodes=1
#SBATCH --output=met_transport_variable_start_%a.out
#SBATCH --array=1-39

# Preamble (get start time, print SLURM env variables, and exit on error)
start=$(date +%s)
print_job_info
set -e

spack env activate /home/AD.UNLV.EDU/perzan/software/spack/env/min3p
 
START_DATE=$(head -n $SLURM_ARRAY_TASK_ID start_dates.txt | tail -n 1)
 
echo "Running $START_DATE"
cd $START_DATE

all_sites=("Cecil" "Flanagan" "HoustonBlack" "Kalamazoo" "Kuma" "Palouse" "Pullman" "Yolo")
follow_up_sims=("monthly" "daily" "hourly")

# Ensure echo is teed into a log file
log_file="$(pwd)"/"$START_DATE.log"
rm -f "$log_file"

# Check that MIN3P_EXEC is set (usually via .bashrc or .load_min3p.sh)
if [[ $MIN3P_EXEC == "" ]]
then
    echo "ERROR -- MIN3P_EXEC is not set." | tee -a "$log_file"
    echo "         Double check .bashrc or .load_min3p.sh. Exiting." | tee -a "$log_file"
    exit
fi

echo "Using $MIN3P_EXEC" | tee -a "$log_file"

for site in "${all_sites[@]}"
do
    pushd "$site"/spinup > /dev/null

    # Get the run name from root.dat
    prefix=$(cat root.dat)

    echo "    Running $site...." | tee -a "$log_file"

    # Run MIN3P and redirect stdout to a log file
    # Also filter out floating-point exceptions from stderr
    $MIN3P_EXEC > "$prefix.shlog" 2> >(grep -v "floating-point exceptions" >&2)

    # If spinup is successful, move the copy the last output file to the next stage
    if tail -n 10 "$prefix.log" | grep -q "normal exit"
    then
        echo "    Spinup successful for $site" | tee -a "$log_file"
        LAST_GSP=$(ls *gsp | sort --version-sort | tail -n 1)
        for sim in "${follow_up_sims[@]}"
        do
            cd ../"$sim"
            printf "        Running %s..." $sim | tee -a "$log_file"
            cp ../spinup/$LAST_GSP ./"$sim".ivs

            # Run MIN3P and redirect stdout to a log file
            # Also filter out floating-point exceptions from stderr
            $MIN3P_EXEC > "$sim.shlog" 2> >(grep -v "floating-point exceptions" >&2)

            # Check if the simulation was successful
            if tail -n 10 "$sim.log" | grep -q "normal exit"
            then
                # Print a newline after the "Running ..." message
                echo | tee -a "$log_file"
            else
                grep "timestep" "$sim.log" | tail -n 1 | awk '{printf"failed after %d days.\n",$4}' | tee -a "$log_file"
            fi
        done
        popd > /dev/null
    else
        echo "    Spinup failed for $site. Check $prefix.log for details." | tee -a "$log_file"
        popd > /dev/null
        continue
    fi

done

n_sites="${#all_sites[@]}"
echo "Finished running simulations for $n_sites" | tee -a "$log_file"
print_runtime $start $(date +%s) | tee -a "$log_file"
echo

print_runtime $start $(date +%s) 
