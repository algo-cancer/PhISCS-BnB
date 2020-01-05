#!/bin/bash

#SBATCH -J ILP
#SBATCH --mail-type=ALL
#SBATCH --mail-user=esamath@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH -o /gpfs/home/e/s/esadeqia/Carbonate/Phylogeny_BnB/reports/slurm_outputs/$(date +%Y%m%d%H%M%S)-$(SLURM_JOB_ID)-$(SLURM_JOB_NAME).txt

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $SLURM_JOB_NODELIS
echo ------------------------------------------------------
echo SLURMD_NODENAME=                         $SLURMD_NODENAME
echo SLURM_CLUSTER_NAME=                      $SLURM_CLUSTER_NAME
echo SLURM_CPUS_ON_NODE=                      $SLURM_CPUS_ON_NODE
echo SLURM_GTIDS=                             $SLURM_GTIDS
echo SLURM_JOBID=                             $SLURM_JOBID
echo SLURM_JOB_ACCOUNT=                       $SLURM_JOB_ACCOUNT
echo SLURM_JOB_CPUS_PER_NODE=                 $SLURM_JOB_CPUS_PER_NODE
echo SLURM_JOB_GID=                           $SLURM_JOB_GID
echo SLURM_JOB_ID=                            $SLURM_JOB_ID
echo SLURM_JOB_NAME=                          $SLURM_JOB_NAME
echo SLURM_JOB_NODELIST=                      $SLURM_JOB_NODELIST
echo SLURM_JOB_NUM_NODES=                     $SLURM_JOB_NUM_NODES
echo SLURM_JOB_PARTITION=                     $SLURM_JOB_PARTITION
echo SLURM_JOB_QOS=                           $SLURM_JOB_QOS
echo SLURM_JOB_UID=                           $SLURM_JOB_UID
echo SLURM_JOB_USER=                          $SLURM_JOB_USER
echo SLURM_LOCALID=                           $SLURM_LOCALID
echo SLURM_MEM_PER_NODE=                      $SLURM_MEM_PER_NODE
echo SLURM_NNODES=                            $SLURM_NNODES
echo SLURM_NODEID=                            $SLURM_NODEID
echo SLURM_NODELIST=                          $SLURM_NODELIST
echo SLURM_NODE_ALIASES=                      $SLURM_NODE_ALIASES
echo SLURM_NPROCS=                            $SLURM_NPROCS
echo SLURM_NTASKS=                            $SLURM_NTASKS
echo SLURM_PRIO_PROCESS=                      $SLURM_PRIO_PROCESS
echo SLURM_PROCID=                            $SLURM_PROCID
echo SLURM_SUBMIT_DIR=                        $SLURM_SUBMIT_DIR
echo SLURM_SUBMIT_HOST=                       $SLURM_SUBMIT_HOST
echo SLURM_TASKS_PER_NODE=                    $SLURM_TASKS_PER_NODE
echo SLURM_TASK_PID=                          $SLURM_TASK_PID
echo SLURM_TOPOLOGY_ADDR=                     $SLURM_TOPOLOGY_ADDR
echo SLURM_TOPOLOGY_ADDR_PATTERN=             $SLURM_TOPOLOGY_ADDR_PATTERN
echo SLURM_WORKING_CLUSTER=                   $SLURM_WORKING_CLUSTER
echo SLURM: SLURM is running on $SLURM_SUBMIT_HOST
echo SLURM: working directory is $SLURM_SUBMIT_DIR
echo SLURM: job identifier is $SLURM_JOB_ID
echo SLURM: job name is $SLURM_JOB_NAME
echo SLURM: NTASK is $SLURM_NTASKS
echo ------------------------------------------------------

source activate env3.6
cd /gpfs/home/e/s/esadeqia/Carbonate/Phylogeny_BnB

python cmp_algs.py --help
