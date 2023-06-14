#!/bin/bash
#SBATCH --job-name=model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH -p high 
#SBATCH --gres=gpu:1  
#SBATCH --time=3-20:00:00
source activate torch-p310 # activate appropriate conda environment
script=${script:-scripts/train_downstream.py}
config_path=${config_path:-/conf}
config_name=${config_name:-config}
split=${split:-1}

# set named arguments
while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi
  shift
done

echo "Which device"
echo $CUDA_VISIBLE_DEVICES

echo 'Script used:' $script
echo 'Config path' $config_path'/'$config_name
#config_dir=bend/conf # relative path to default conf dir
HYDRA_FULL_ERROR=1 python -u $script --config-path $config_path --config-name $config_name  #--config-dir $config_dir #--info serachpath ++data.partition_split=1 #params.mode=test ++data.partition_split=$split
