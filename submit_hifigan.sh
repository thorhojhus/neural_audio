#!/bin/sh
#BSUB -q gpua100
#BSUB -R "select[gpu40gb]"
#BSUB -J train_descript_audiocodec
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 03:00
#BSUB -R "rusage[mem=16GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

module swap python3/3.10.11
module swap cuda/12.1

python3 /zhome/5c/8/186818/deep_learning_audio_project/training_hifiplusplus_submit.py