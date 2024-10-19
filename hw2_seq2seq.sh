#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./hw2_seq2seq.sh <data_directory> <output_filename>"
    exit 1
fi

# Assign the command-line arguments to variables
DATA_DIR=$1
OUTPUT_FILE=$2

# Model parameters and directories
MODEL_SAVE_DIR="saved_models"
MODEL_SAVE_PATH="${MODEL_SAVE_DIR}/video_captioning_model.pth"

# Create the model directory if it doesn't exist
mkdir -p ${MODEL_SAVE_DIR}

# Run the Python script with the provided arguments
python video_captioning.py --data_dir $DATA_DIR --output_file $OUTPUT_FILE

"

echo "Inference completed. Results saved to ${OUTPUT_FILE}"
