# Path to checkpoint file or a directory containing checkpoint files. Passing
# a directory will only work if there is also a file named 'checkpoint' which
# lists the available checkpoints in the directory. It will not work if you
# point to a directory with just a copy of a model checkpoint: in that case,
# you will need to pass the checkpoint path explicitly.
CHECKPOINT_PATH="${HOME}/proj_bach/im2txt_LSTM/im2txt/model/train_12h"

# Vocabulary file generated by the preprocessing script.
VOCAB_FILE="/mnt/d/data/mscoco/word_counts.txt"

# JPEG image file to caption.
IMAGE_FILE="${HOME}/proj_bach/achien.jpg"

# Build the inference binary.
cd ${HOME}/proj_bach/im2txt_LSTM/
bazel build -c opt //im2txt:run_inference

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=""

# Run inference to generate captions.
bazel-bin/im2txt/run_inference \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE}