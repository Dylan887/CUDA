#!/bin/bash

# Define color codes
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
CYAN='\033[1;36m'
RESET='\033[0m'  # Reset color

# Print usage function
usage() {
    echo -e "${YELLOW}Usage:${RESET} $0 {train|inference} [train|inference]"
    echo -e "${YELLOW}Example:${RESET} ./start.sh train"
    echo -e "         ./start.sh inference"
    echo -e "         ./start.sh train inference"
    exit 1
}

# Check if at least one argument is provided
if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
fi

# Parse arguments
DO_TRAIN=false
DO_INFERENCE=false

for arg in "$@"; do
    if [[ "$arg" == "train" ]]; then
        DO_TRAIN=true
    elif [[ "$arg" == "inference" ]]; then
        DO_INFERENCE=true
    else
        echo -e "${RED}Invalid argument:${RESET} $arg"
        usage
    fi
done

# Define performance metrics
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

# Print current working directory with color
echo -e "${CYAN}Current directory: $(pwd)${RESET}"

# Perform training if requested
if [[ "$DO_TRAIN" == true ]]; then
    if [ -f "train.py" ]; then
        echo -e "${MAGENTA}=============================================${RESET}"
        echo -e "${GREEN}TRAINING PERFORMANCE TEST IS STARTING...${RESET}"
        echo -e "${MAGENTA}=============================================${RESET}"
        
        # Run NSYS profiling
        nsys profile --trace=cuda,cudnn,nvtx,osrt --output=train python train.py
        if [ -f "train.nsys-rep" ]; then
            echo -e "${YELLOW}Processing training performance metrics...${RESET}"
            for metric in "${metrics[@]}"; do
                nsys stats "train.nsys-rep" -r "$metric" --format csv --output=train
            done
            echo -e "${GREEN}Training performance test completed successfully!${RESET}"
        else
            echo -e "${RED}train.nsys-rep not found. Training performance test failed!${RESET}"
        fi
    else
        echo -e "${RED}train.py not found in the current directory. Skipping training.${RESET}"
    fi
fi

# Perform inference if requested
if [[ "$DO_INFERENCE" == true ]]; then
    if [ -f "inference.py" ]; then
        echo -e "${MAGENTA}=============================================${RESET}"
        echo -e "${GREEN}INFERENCE PERFORMANCE TEST IS STARTING...${RESET}"
        echo -e "${MAGENTA}=============================================${RESET}"
        
        # Run NSYS profiling
        nsys profile --trace=cuda,cudnn,nvtx,osrt --output=inference python inference.py
        if [ -f "inference.nsys-rep" ]; then
            echo -e "${YELLOW}Processing inference performance metrics...${RESET}"
            for metric in "${metrics[@]}"; do
                nsys stats "inference.nsys-rep" -r "$metric" --force-export=true --format csv --output=inference
            done
            echo -e "${GREEN}Inference performance test completed successfully!${RESET}"
        else
            echo -e "${RED}inference.nsys-rep not found. Inference performance test failed!${RESET}"
        fi
    else
        echo -e "${RED}inference.py not found in the current directory. Skipping inference.${RESET}"
    fi
fi

echo -e "${CYAN}Performance test completed.${RESET}"
