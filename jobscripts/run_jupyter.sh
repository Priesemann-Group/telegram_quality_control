#!/bin/bash

#SBATCH -p nariman
#SBATCH --job-name=export_data
#SBATCH --output=./logs/%x_%j.log
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=480GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anastasia.golovin@ds.mpg.de

WORKSPACE_DIR="/data/nst/agolovin/projects/telegram_research/quality_control"

cd $WORKSPACE_DIR

eval $(poetry env activate)

# Function to convert Jupyter notebook to Python script
run_jupyter_notebook() {
    local input_notebook="$1"
    local output_script="$2"
    
    echo "Converting notebook to Python script..."
    echo "  Input: $input_notebook"
    echo "  Output: $output_script"
    
    jupyter nbconvert --to python "$input_notebook" --output="$output_script"
    
    if [ $? -ne 0 ]; then
        echo "Error: Notebook conversion failed"
        return 1
    fi
    
    echo "Conversion completed successfully"

    # Execute the Python script
    echo "Executing Python script..."
    python3 "$PYTHON_SCRIPT"

    if [ $? -ne 0 ]; then
        echo "Error: Python script execution failed"
        exit 1
    fi
}

INPUT_NOTEBOOK="$WORKSPACE_DIR/notebooks/data/export_data.ipynb"
PYTHON_SCRIPT="$WORKSPACE_DIR/notebooks/data/export_data.py"

run_jupyter_notebook "$INPUT_NOTEBOOK" "$PYTHON_SCRIPT"

echo "Job completed successfully at $(date)"

