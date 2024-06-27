#!/bin/bash

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_satfeatpy

# Define directories
cnf_dir="cnf_examples"
results_dir="results"
sub_dir="more_complex_cnfs"

# Ensure results directory exists
mkdir -p $results_dir

# Function to run profiling
run_profiling() {
    branch=$1
    file=$2
    filename=$(basename -- "$file")
    filename="${filename%.*}"

    # Define profile and output files based on the branch
    if [ "$branch" == "main" ]; then
        profile_file="$results_dir/${filename}.original.prof"
        output_file="$results_dir/${filename}.original.output"
    else
        profile_file="$results_dir/${filename}.optimized.prof"
        output_file="$results_dir/${filename}.optimized.output"
    fi

    # Checkout the branch
    git checkout $branch

    # Run the profiling
    python -m cProfile -o $profile_file generate_features.py $file > $output_file
}

# Iterate through all .cnf files in the cnf_examples folder and its subfolder
for file in $cnf_dir/*.cnf $cnf_dir/$sub_dir/*.cnf; do
    if [ -f "$file" ]; then
        # Run profiling on main branch
        run_profiling "main" "$file"
        
        # Run profiling on aashna09/optimization branch
        run_profiling "aashna09/optimization" "$file"
    fi
done

# Checkout back to the main branch
git checkout main

# Deactivate the conda environment
conda deactivate
