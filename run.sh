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

# Pretrained model
PRETRAINED_MODEL_NAME_OR_PATH="openai/clip-vit-base-patch32"

# Dataset
DATASET="clip"
DATASET_NAME="../local_pickascore"
DATASET_CONFIG="default"
FROM_DISK=true
BATCH_SIZE=2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    dataset.batch_size=$BATCH_SIZE \
    model.pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH \
    task.pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH \
    dataset.processor.pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH \