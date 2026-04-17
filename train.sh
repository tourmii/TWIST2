#!/bin/bash

# Usage: bash train.sh <experiment_id> <device>

# bash train.sh 1103_twist2 cuda:0

export WANDB_API_KEY="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"
export WANDB_MODE="run"

cd legged_gym/legged_gym/scripts

robot_name="g1"
exptid=$1
device=$2

task_name="${robot_name}_mimic"
proj_name="${robot_name}_mimic"


# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "None" \
                # --resume \
                # --debug \
