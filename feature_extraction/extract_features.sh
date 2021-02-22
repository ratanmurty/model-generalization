#!/bin/sh
#SBATCH --mem=256GB
#SBATCH -t 72:0:0

tf_slim_path=/mindhive/nklab4/users/ighodgao
script_path=.

source ~/.bashrc
conda activate "brain-score"
export PYTHONPATH="$PYTHONPATH:$tf_slim_path/tf-models/research/slim"
cd $script_path
python3 extract_features.py --images_path=$1 --model=$2 --save_dir=$3 --index=$4
