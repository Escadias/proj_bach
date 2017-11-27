#!/bin/bash

#SBATCH -J train_model
#SBATCH -o train_model.o%j
#SBATCH --constraint="V5|V6"
#SBATCH --partition=shared-gpu
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:titan:2
#SBATCH --mem=24GB

module load GCC/4.9.3-2.25 OpenMPI/1.10.2 Python/2.7.11 foss/2016a Java/1.8.0_92 CUDA cuDNN tensorflow/1.0.1-Python-2.7.11

# if you need to know the allocated CUDA device, you can obtain it here:
echo $CUDA_VISIBLE_DEVICES

cd ~/proj_bach/im2txt
source python_virtual_env/bin/activate
/opt/bazel/bazel-0.4.4/bazel build -c opt //im2txt/...

bazel-bin/im2txt/train \
  --input_file_pattern=im2txt/data/mscoco/train-?????-of-00256 \
  --inception_checkpoint_file=im2txt/data/inception_v3.ckpt \
  --train_dir=im2txt/model/train \
  --train_inception=false \
  --number_of_steps=100
