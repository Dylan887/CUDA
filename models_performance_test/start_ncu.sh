#!bin/bash

#define color codes

RED='\033[1:31m'
GREEN='\033[1:32m'
YELLOW='\033[1:33m'
BLUE='\033[1:34m'
MAGENTA='\033[1:35m'
CYAN='\033[1:36m'
RESET='\033[0m'

#usage function
usage() {
    echo -e "${YELLOW}Usage:${RESET} $0 {train|inference} [train|inference]"
    echo -e "${YELLOW}Example:${RESET} ./start_ncu.sh train"
    echo -e "         ./start_ncu.sh inference"
    echo -e "         ./start_ncu.sh train inference"
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

# Print current working directory with color
echo -e "${CYAN}Current directory: $(pwd)${RESET}"

# Perform training if requested
if [[ "$DO_TRAIN" == true ]]; then
    if [ -f "train.py" ]; then
        echo -e "${MAGENTA}=============================================${RESET}"
        echo -e "${GREEN}NCU performance test for training is starting...${RESET}"
        echo -e "${MAGENTA}=============================================${RESET}"
        
        # Run NCU profiling
        ncu --set full --export train.ncu-rep --target-processes auto --replay-mode application python train.py
        if [ -f "train.ncu-rep" ]; then
            echo -e "${GREEN}NCU performance test for training completed successfully!${RESET}"
        else
            echo -e "${RED}train.ncu-rep not found. Training performance test failed!${RESET}"
        fi
    else
        echo -e "${RED}train.py not found in the current directory. Skipping training.${RESET}"
    fi
fi

# Perform inference if requested
if [[ "$DO_INFERENCE" == true ]]; then
    if [ -f "inference.py" ]; then
        echo -e "${MAGENTA}=============================================${RESET}"
        echo -e "${GREEN}NCU performance test for inference is starting...${RESET}"
        echo -e "${MAGENTA}=============================================${RESET}"
        # do train before inference
        if find . -maxdepth 1 -name "*.pth" -print -quit | grep -q .; then
            # Run NCU profiling
            ncu --set full --export inference.ncu-rep --target-processes auto --replay-mode application python train.py
            if [ -f "inference.ncu-rep" ]; then
                echo -e "${GREEN}NCU performance test for inference completed successfully!${RESET}"
            else
                echo -e "${RED}inference.ncu-rep not found. Inference performance test failed!${RESET}"
            fi
        elif 
            python train.py
            ncu --set full --export inference.ncu-rep --target-processes auto --replay-mode application python train.py
            if [ -f "inference.ncu-rep" ]; then
                echo -e "${GREEN}NCU performance test for inference completed successfully!${RESET}"
            else
                echo -e "${RED}inference.ncu-rep not found. Inference performance test failed!${RESET}"
            fi

        fi
    else
        echo -e "${RED}inference.py not found in the current directory. Skipping inference.${RESET}"
    fi
fi

echo -e "${CYAN}NCU performance test completed.${RESET}"

