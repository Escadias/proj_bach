#!/bin/bash

#SBATCH -J evaluation_LSTM
#SBATCH -o output/evaluation_LSTM.o%j
#SBATCH --constraint="V5|V6"
#SBATCH --partition=shared-gpu
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:titan:2
#SBATCH --mem=24GB

module purge
module load GCC/4.9.3-2.25 OpenMPI/1.10.2 Python/2.7.11 foss/2016a Java/1.8.0_92 cuDNN/5.1-CUDA-8.0.44 tensorflow/1.3.0-Python-2.7.11

# if you need to know the allocated CUDA device, you can obtain it here:
echo $CUDA_VISIBLE_DEVICES

cd ~/proj_bach/
source python_virtual_env/bin/activate
cd im2txt_LSTM

/opt/bazel/bazel-0.4.4/bazel build -c opt //im2txt/evaluate

# Run the evaluation script. This will run in a loop, periodically loading the
# latest model checkpoint file and computing evaluation metrics.
bazel-bin/im2txt/evaluate \
  --input_file_pattern=$HOME/proj_bach/data/mscoco/val-?????-of-00004 \
  --checkpoint_dir=im2txt/model/train_12h \
  --eval_dir=im2txt/model/eval_12h