#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

# Edited by Adam Scicluna for RPV pipeline, 2026

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)

export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export SAM3_CHECKPOINT=${SAM3_CHECKPOINT:-checkpoints/sam3.pt}
#export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export YOLOE_CHECKPOINT=${YOLOE_CHECKPOINT:-checkpoints/yoloe-26x-seg.pt}   # If prompt-free, use yoloe-26x-seg-pf.pt; if prompt-based, use yoloe-26x-seg.pt
export MASK2FORMER_CHECKPOINT=${MASK2FORMER_CHECKPOINT:-Mask2Former/checkpoints/ade20k-semseg-r50_model_final_500878.pkl}
export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}
export MASK2FORMER_PORT=${MASK2FORMER_PORT:-12181}
export CLIP_PORT=${CLIP_PORT:-12182}
export SAM3_PORT=${SAM3_PORT:-12183}
export YOLOE_PORT=${YOLOE_PORT:-12184}

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically
tmux split-window -v -t ${session_name}:0

# Split both panes horizontally
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 "${VLFM_PYTHON} -m vlfm.vlm.mask2former --port ${MASK2FORMER_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "${VLFM_PYTHON} -m vlfm.vlm.clip --port ${CLIP_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "${VLFM_PYTHON} -m vlfm.vlm.sam3 --port ${SAM3_PORT}" C-m
tmux send-keys -t ${session_name}:0.3 "${VLFM_PYTHON} -m vlfm.vlm.yoloe --port ${YOLOE_PORT}" C-m

# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'. Please wait for the model weights to finish being loaded before starting evaluation."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
