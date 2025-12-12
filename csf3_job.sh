#!/bin/bash --login
#$ -cwd
#$ -l a100           
# A 1-GPU request (v100 is just a shorter name for nvidia_v100)
# Can instead use 'a100' for the A100 GPUs (if permitted!)

#$ -pe smp.pe 8      # 8 CPU cores available to the host code
# Can use up to 12 CPUs with an A100 GPU.

# Latest version of CUDA
module load libs/cuda

echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"
singularity exec --nv --no-home docker://mingfeisun/procgen:pytorch bash train.sh