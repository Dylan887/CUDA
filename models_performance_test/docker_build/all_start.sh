#!/bin/bash

# Define color codes
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
CYAN='\033[1;36m'
RESET='\033[0m'  # Reset color

# Model directory path
MODEL_DIR="/workspace/models"
models=$(ls "$MODEL_DIR" | sort -V)

# Define an array of performance metrics
metrics=(
    "cuda_api_gpu_sum"
    "cuda_api_sum"
    "cuda_api_trace"
    "cuda_gpu_kern_gb_sum"
    "cuda_gpu_kern_sum"
    "cuda_gpu_mem_size_sum"
    "cuda_gpu_mem_time_sum"
    "cuda_gpu_sum"
    "cuda_gpu_trace"
    "cuda_kern_exec_sum"
    "cuda_kern_exec_trace"
    "osrt_sum"
    "um_cpu_page_faults_sum"
    "um_sum"
    "um_total_sum"
)

# Loop through the model directory
for model_n in $(ls "$MODEL_DIR" | sort -V); do
    model="$MODEL_DIR/$model_n"
    # Skip the 'data' and 'pre_weights' directories
    if [[ "$model" == *"/data"* || "$model" == *"/pre_weights"* ]]; then
        echo -e "${YELLOW}Skipping directory: $model${RESET}"
        continue
    fi

    if [ -d "$model" ]; then
        model_name=$(basename "$model")ls


        cd "$model" || { echo -e "${RED}Failed to change directory to $model${RESET}"; continue; }
        echo -e "${MAGENTA}=============================================${RESET}"
        echo -e "${CYAN}${model_name} PERFORMANCE TEST IS STARTING...${RESET}"
        echo -e "${MAGENTA}=============================================${RESET}"
        
        if [ -f "train.py" ]; then
            echo -e "${YELLOW}Profiling training...${RESET}"
            nsys profile --trace=cuda,osrt --output=train python train.py
            if [ -f "train.nsys-rep" ]; then
                for metric in "${metrics[@]}"; do
                    nsys stats "train.nsys-rep" -r "$metric" --force-export=true --format csv --output=train
                done
                echo -e "${GREEN}${model_name} training performance test completed successfully!${RESET}"
            else
                echo -e "${RED}train.nsys-rep not found. Training performance test failed!${RESET}"
                echo -e "${RED}${model_name} training performance test is wrong!${RESET}"
                continue
            fi
        else
            echo -e "${RED}train.py not found in the current directory. Stopping.${RESET}"
            exit 1
        fi

        if ls *.pth &>/dev/null; then
            if [ -f "inference.py" ]; then
                echo -e "${YELLOW}Profiling inference...${RESET}"
                nsys profile --trace=cuda,osrt --output=inference python inference.py
                if [ -f "inference.nsys-rep" ]; then
                    for metric in "${metrics[@]}"; do
                        nsys stats "inference.nsys-rep" -r "$metric" --force-export=true --format csv --output=inference
                    done
                    echo -e "${GREEN}${model_name} inference performance test completed successfully!${RESET}"
                else
                    echo -e "${RED}inference.nsys-rep not found. Inference performance test failed!${RESET}"
                    echo -e "${RED}${model_name} inference performance test is wrong!${RESET}"
                    continue
                fi
            else
                echo -e "${RED}inference.py not found in the current directory. Stopping.${RESET}"
                exit 1
            fi
        else
            echo -e "${RED}No .pth files found. Stopping.${RESET}"
            exit 1
        fi
        echo -e "${GREEN}${model_name} performance test finished!${RESET}"
    fi

    echo -e "${BLUE}Waiting for GPU resources to be released...${RESET}"
    sleep 6
done

echo -e "${CYAN}All models processed.${RESET}"
