#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=8g

module load tensorflow/1.12.0-py36-gpu
module load nccl
module load openmpi

export PYTHONPATH=$PYTHONPATH:/flush1/wu082/proj/claran/pyenv
#source /flush1/wu082/venvs/claran_env/bin/activate
LOG_DIR=/flush1/wu082/proj/claran/train_log/$SLURM_JOB_ID
cd /flush1/wu082/proj/claran
mpirun -np 1 python train.py --logdir $LOG_DIR --config \
        MODE_MASK=False MODE_FPN=True \
        DATA.BASEDIR=./data \
        BACKBONE.WEIGHTS=./weights/pretrained/ImageNet-R50-AlignPadding.npz \
        DATA.TRAIN=trainD1 DATA.VAL=testD1 \
        PREPROC.TRAIN_SHORT_EDGE_SIZE=600,600 \
        PREPROC.TEST_SHORT_EDGE_SIZE=600 \
	TRAIN.LR_SCHEDULE=20000,30000,40000
