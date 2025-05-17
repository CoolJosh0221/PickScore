# export PYTHONPATH="$(pwd)/.."

######
# Configuration
######

# Accelerate settings
DYNAMO_BACKEND="no"
GPU_IDS="all"
NUM_MACHINES=1
NUM_PROCESSES=1
MIXED_PRECISION="no"
USE_DEEPSPEED="yes"

# Experiment & I/O
EXPERIMENT="clip_h"
OUTPUT_DIR="outputs"

# Dataset
DATASET="clip"
DATASET_NAME="../local_pickascore"
DATASET_CONFIG="default"
FROM_DISK=true
BATCH_SIZE=4

######
# Launch!
######
PYTHONPATH=$PWD accelerate launch \
  --dynamo_backend $DYNAMO_BACKEND \
  --gpu_ids $GPU_IDS \
  --num_machines $NUM_MACHINES \
  --num_processes $NUM_PROCESSES \
  --mixed_precision $MIXED_PRECISION \
  $( [ "$USE_DEEPSPEED" = "yes" ] && echo "--use_deepspeed" ) \
  trainer/scripts/train.py \
    +experiment=$EXPERIMENT \
    output_dir=$OUTPUT_DIR \
    dataset=$DATASET \
    dataset.dataset_name="$DATASET_NAME" \
    dataset.dataset_config_name="$DATASET_CONFIG" \
    dataset.from_disk=$FROM_DISK \
    dataset.batch_size=$BATCH_SIZE