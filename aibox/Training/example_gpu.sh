#!/bin/bash

## set the maximum time your job will run for (the absolute maximum is 48 hours)
#SBATCH --time=12:00:00 

## set the number of nodes being requested (nearly always 1)
#SBATCH --nodes 1  

## set the number of tasks per node (usually 1) 
#SBATCH --ntasks-per-node=1

## set the amount of RAM you're requesting 
#SBATCH --mem 75G

## set the number of CPUs you're requesting (max is 10)
#SBATCH -c 10

## set the partition you want to submit your job to, here the GPUs
#SBATCH -p gpu

## set the GPU you want to use
#SBATCH --gres=gpu:H100.10gb:1 

## name the error file as desired
#SBATCH --error=error_nc.o%j

## name the output file as desired
#SBATCH --output=output_nc.o%j

## if you want your job to run only after another job has run, remove one # and enter that job's ID here (which you can get by running squeue -u your_username)
##SBATCH --dependency=afterok:(job ID) ## without the parentheses, ex afterok:123456

## add your university email so the system can email you updates on your job
#SBATCH --mail-user=ppowell@uni-osnabrueck.de

## remove one # from this if you want the system to email you when the job starts
##SBATCH --mail-type=BEGIN

## remove one # from this if you want the system to email you when the job ends
##SBATCH --mail-type=END

## remove one # from this if you want the system to email you if your job fails (encounters an error)
##SBATCH --mail-type=FAIL

## remove one # from this if you want the system to email you for all events
##SBATCH --mail-type=ALL

## These are just general commands
echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo

## load miniconda, activate your environment, and set the TMPDIR to your folder on the shared drive in case your program creates any temporary files
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate (your environment)
export TMPDIR='/share/neurobiopsychologie/(yourusername)'

## run your program
srun python /full/path/to/yourfile.py