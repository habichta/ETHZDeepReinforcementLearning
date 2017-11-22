#!/bin/bash

#
##SBATCH --job-name="knet"
##SBATCH --output=test.out
##SBATCH --error=test.out
#
#
#
# run on all cores minus one of the node, require 2GB per core = 14GB
#SBATCH --partition=plongx
#SBATCH --nodes=1
#
# make the submitter happy and print what is been done
tstamp="`date '+%D %T'`"
hn="`hostname -f`"
jobid=${SLURM_JOB_ID}
jobname=${SLURM_JOB_NAME}
if [ -z "${jobid}" ] ; then
  echo "ERROR: SLURM_JOBID undefined, are you running this script directly ?"
  exit 1
fi

printf "%s: starting %s(%s) on host %s\n" "${tstamp}" "${jobname}" "${jobid}" "${hn}"
echo "**"
echo "** SLURM_CLUSTER_NAME="$SLURM_CLUSTER_NAME
echo "** SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "** SLURM_JOB_ID="$SLURM_JOBID
echo "** SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "** SLURM_NUM_NODES"=$SLURM_NUM_NODES
echo "** SLURMTMPDIR="$SLURMTMPDIR
echo "** working directory = "$SLURM_SUBMIT_DIR
echo
echo "Loading virtual environment:"
source ~/keras/bin/activate
echo "loading modules:"
module load cuda/8.0.27
module load cudnn/v5.1
echo "Starting job"
hn="`hostname -f`"
# echo "starting job with NPROCS=$NPROCS:"
dt="`date '+%s'`"


python /home/pdinesh/knet-euryale/models/vgg_unseendata.py /home/pdinesh/knet-euryale/models/params_euryale.json
#python /home/pdinesh/git/keras/examples/mnist_cnn.py

stat="$?"
dt=$(( `date '+%s'` - ${dt} ))
echo "job finished, status=$stat, duration=$dt second(s)"
echo


