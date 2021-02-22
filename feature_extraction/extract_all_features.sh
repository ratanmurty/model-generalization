#!/bin/sh
num_base_models=93
num_cornets=15
num_pytorch_models=12
dataset_path=$1
save_path_base=$2

for ((x=0; x<$num_base_models; x++))
do
	echo $x
	sbatch extract_features.sh $dataset_path "base-models" $save_path_base"brain-score" $x	

done

for ((x=0; x<$num_cornets; x++))
do
        echo $x
        sbatch extract_features.sh $dataset_path "cornets" $save_path_base"pytorch" $x        
done

for ((x=0; x<$num_pytorch_models; x++))
do
        echo $x
        sbatch extract_features.sh $dataset_path "pytorch" $save_path_base"pytorch" $x        
done



