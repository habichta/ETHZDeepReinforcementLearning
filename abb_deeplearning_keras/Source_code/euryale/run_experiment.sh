#!/bin/bash



node=$1;
exp_name=$2;

if [ -z "${node}" ] ; then
  echo "ERROR: GPU_NODE / EXPERIMENT_NAME undefined, are you running this script directly ?"
  exit 1
fi

if [ -z "${exp_name}" ] ; then
  echo "ERROR: GPU_NODE / EXPERIMENT_NAME undefined, are you running this script directly ?"
  exit 1
fi


echo "Initializing experiment.. "$exp_name
echo "Creating a new job on Node "$node


euryale='/home/pdinesh/knet-euryale/euryale'
models='/home/pdinesh/knet-euryale/models/'
exp_logs='/home/pdinesh/exp_logs/'

echo "Saving experiment parameters to disk"
cp /home/pdinesh/knet-euryale/models/params_euryale.json /home/pdinesh/exp_logs/$exp_name.json


sbatch --nodelist=node[$node] --gres="gpu" --job-name=$exp_name --output=$exp_logs$exp_name".out" --error=$exp_logs$exp_name".out" $euryale"/"sbatch.sh

echo "Job is queued!"
