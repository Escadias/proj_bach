MSCOCO_DIR="/mnt/d/data/mscoco"
MODEL_DIR="${HOME}/proj_bach/im2txt_GRU/im2txt/model"

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
#export CUDA_VISIBLE_DEVICES=""

bazel build -c opt //im2txt/...


# Run the evaluation script. This will run in a loop, periodically loading the
# latest model checkpoint file and computing evaluation metrics.
bazel-bin/im2txt/evaluate \
  --input_file_pattern="${MSCOCO_DIR}/val-?????-of-00004" \
  --checkpoint_dir="${MODEL_DIR}/train_12h" \
  --eval_dir="${MODEL_DIR}/eval_12h"
