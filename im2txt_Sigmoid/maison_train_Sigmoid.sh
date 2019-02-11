# Directory containing preprocessed MSCOCO data.
MSCOCO_DIR="/mnt/d/data/mscoco"

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="/mnt/d/data/inception_chkpt/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="${HOME}/proj_bach/im2txt_Sigmoid/im2txt/model"


# Build the model.
cd ${HOME}/proj_bach/im2txt_Sigmoid/
bazel build -c opt //im2txt/...

# Run the training script.
bazel-bin/im2txt/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=3 #1000000
