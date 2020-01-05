#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00
#PBS -l vmem=16gb

#PBS -N ILP
#PBS -V
#PBS -M esamath@gmail.com
#PBS -m abe 
#PBS -j oe 
echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo ------------------------------------------------------

module unload python/2.7.16
module load anaconda/python3.6/4.3.1

source activate env2
cd /gpfs/home/e/s/esadeqia/Carbonate/Phylogeny_BnB
pwd

python cmp_algs.py --help
