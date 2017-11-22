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


python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-01-23 2016-01-23
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-01-24 2016-01-24
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-01-25 2016-01-25
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-01-26 2016-01-26
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-01-27 2016-01-27
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-01-28 2016-01-28
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-01-29 2016-01-29
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-01-30 2016-01-30
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-01-31 2016-01-31
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-02-01 2016-02-01
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-02-02 2016-02-02
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-02-03 2016-02-03
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-02-04 2016-02-04
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-02-05 2016-02-05
python /home/pdinesh/knet-euryale/models/predict_daily_cav_2cls.py /home/pdinesh/knet-euryale/models/params_euryale.json 2016-02-06 2016-02-06

stat="$?"
dt=$(( `date '+%s'` - ${dt} ))
echo "job finished, status=$stat, duration=$dt second(s)"
echo


