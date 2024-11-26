#!/bin/bash

# Path to the model directory
MODEL_DIR="/workspace/models"

# Define the data directory to exclude
DATA_DIR="$MODEL_DIR/data"
DATA_DIR2="$MODEL_DIR/pre_weights"

# Iterate through each model directory
for model in "$MODEL_DIR"/*; do
    if [ -d "$model" ]; then
        echo "Processing model directory: $model"

        # Define the files to keep
        train_file="$model/train.py"
        inference_file="$model/inference.py"

        # Loop through all files and subdirectories in the model directory
        for item in "$model"/*; do
            # If the item is not train.py, inference.py, or the data directory, delete it
            if [ "$item" != "$train_file" ] && [ "$item" != "$inference_file" ] && [ "$item" != "$DATA_DIR" ]&& [ "$item" != "$DATA_DIR2" ]; then
                echo "Deleting: $item"
                rm -rf "$item"
            fi
        done
    fi
done

echo "Cleanup completed."
