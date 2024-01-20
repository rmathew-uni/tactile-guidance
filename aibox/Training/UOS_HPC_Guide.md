Uni-Osnabrück HPC Guide

The following guide will help you get started on the university's high-performance computing network. Note that the HPC is entirely terminal-based and has no GUI. 

// Resources //

First, the basics of what hardware you're accessing and how things are laid out:

The main section of the HPC which you will be using contains:
- 51 CPU nodes, each containing 128 CPU cores and 1TB of memory 
- 2 GPU nodes, each containing 128 CPU cores, 1TB of memory, and 4 NVIDIA A100 GPUs 

On the HPC, you have a home directory under /home/student/(first letter of your last name)/(your username)/. Programs and data are stored in the /share/ directory. In the case of the NBP, our data and project folders can be found in /share/neurobiopsychologie/. To store code you will be using in your personal work, create a directory named with your username within the neurobiopsychologie directory (using the mkdir command). Projects with multiple members have their own directories (when working in these, please refer to the section of this guide dealing with permissions to ensure that your teammates can access any files you edit or create in the project directories). 

Note that the share directory has no backup, so it is advisable to maintain a copy of your code elsewhere as well. All nodes on the HPC have access to the internet so you can use git to maintain your code in an external repo if desired.

// Accessing the HPC //

You access the HPC by logging into the master management login node - HPC3. This is achieved via an ssh connection when connected to the university network either via eduroam or via the university's VPN. So, as follows:

- In a terminal, type the following command: ssh (network username)@hpc3.rz.uos.de
- You will then be prompted for your network password (same as when you log into your email). You have 3 tries to enter this correctly before the node kicks you off and you have to ssh back in again. 
- Note that it is also possible to set up an ssh key to allow for passwordless connection between your laptop and the network, exactly like when you access a project hosted on GitHub/GitLab, but this is not covered in this guide. 

Once you log in, a startup text block will be printed in your terminal (begins with "Dieses Cluster wurde durch die DFG finanziert..."). Once you see this, you have successfully logged into the login node of the HPC cluster. Note that this node is not intended for computations! You should NEVER run any programs on this node, although minor operations (directory creation and management, etc.) are fine. 

By default, the log in node will open to your home directory. You can confirm this by running the command ls in the terminal. If you are just starting with the HPC, this directory may be empty. Later, this will usually be where your conda environments are stored. 

// Running Programs on the HPC - Interactive Sessions //

There are two ways to run jobs on the HPC - via a submitted Slurm job or within an interactive session. Typically, you will want to work in an interactive session when debugging or testing code, and to submit a job when you are confident your code is clean and can allow the system to schedule it when the appropriate resources are available. 

You may access either the CPU nodes or the GPU nodes for both jobs and interactive sessions. 

Let's start with how to access an interactive session first:

To request an interactive session on the CPUs, input the following command into your terminal: 

salloc -p workq -n 1 -c 10 --mem 20G srun --pty bash

- salloc is the command requesting that the Slurm management system allocate you resources for your session
- the -p argument specifies which partition you wish to have the interactive session on; in ths case we ask for workq, the CPU partition
- the -n argument specifies how many nodes you wish to have allocated to you; this should almost always be 1
- the -c command specifies how many CPU cores you want allocated to you; this should never be more than 10 
- the --mem command specifies how much RAM you want allocated to you; try not to take more than you need, which you can estimate by evaluating what data your program actively maintains in memory (i.e., do you have growing lists that sit in memory, does it load a dataframe combining multiple individual data files, etc.)
- the srun --pty bash command loads your interactive session with conda already activated and ready to use (this guide covers the creation of conda environments on the HPC later)

To request an interactive session on the GPUs, input the following command into your terminal: 

salloc -p gpu -n 1 -c 10 --mem 20G --gres=gpu:A100:1 srun --pty bash

- in this case, we set the partition to gpu, the name of the GPU partition
- we add the additional argument --gres, which specifies the exact GPU being used, in this case the NVIDIA A100

Once you input the command to request an interactive session, the management system will check if the resources you requested are available. If so, you will see the following:

salloc: Pending job allocation X
salloc: job X queued and waiting for resources
salloc: job X has been allocated resources
salloc: Granted job allocation X
salloc: Waiting for resource configuration
salloc: Nodes X are ready for job

If the resources are not available, you will only see the first two lines and will have to wait until the resources are freed up as other jobs are completed. Because of the high demand on the GPUs, you may have to wait quite a while before the system can grant your request. You will usually be granted an interactive session on the CPUs faster, but during high-demand periods, you may also have to wait for a session on these nodes as well. 

// Python Environments on the HPC //

Whether you are running a program in a job or in an interactive session, you will of course need a Python environment. The HPC already contains multiple packages that will be useful for you when running programs, including miniconda. Using the pre-installed miniconda on the HPC is in fact the recommended way to create and manage environments on the HPC, so let's cover this method:

- run export TMPDIR=/share/neurobiopsychologie/(your username)/ - this will change the temporary directory to your directory on the share drive, crucial when installing many packages at once or very large packages, since the default temporary folder is rather small (if you forget this step, you will get an OS memory error when installing the package(s)) 
- load an interactive session on a CPU node using the method above (this will ensure you have sufficient memory for installing large packages)
- run spack load miniconda3
- run conda create --name (name) to create a new environment accessible from the root with the given name
- if you have a requirements.txt file or an environment.yml file that lists the packages you want installed in your new environment, follow the following steps:
- say you have a requirements.txt file in /share/neurobiopsychologie/(your username)/
- first run the command cd /share/neurobiopyschologie/(your username) command to change to the directory with the environment file
- run conda env create -f requirements.txt to create the environment with the environment file (note the addition of env before create!)

With your environment created, you can now use it in an interactive session or in a job. Let's start with an interactive session:
- request an interactive session with the steps above
- run spack load miniconda3
- run conda activate (your environment)
- cd to the directory with the program you want to run
- run python your_file.py to run the program 
- the program will run live in your terminal

// Running Programs on the HPC - Standard Submission //

Now let's cover submitting a job, which is how you'll usually run your programs. To do this, you first need to create an .sh file, which lists the parameters and commands for your job (essentially what you put in when you request and use an interactive session). An example .sh file (example_cpu.sh) can be found in /share/neurobiopsychologie/ and demonstrates how to structure this file for submitting a job to the CPU nodes. The file example_gpu.sh demonstrates the same for submitting to the GPU nodes. 

Your .sh job file is usually placed in the same directory as the program you want to run. 

To submit your job, cd to the directory where your .sh file is located and enter the command sbatch yourfile.sh, which will submit your job to the Slurm management system. You will receive a confirmation printout that your job has been submitted which includes the job number it was assigned.

Once your job has been submitted, you can monitor its progress in the Slurm system by running the command squeue -u (your username). This will list all jobs you currently have running, along with their job number, the partition they were submitted to, the amount of time they have been running, and their current status and the reason for that status. The following statuses are possible:

- R: your job is currently running, the time counter will tell you how long it has been running for
- PD: your job is currently pending, usually because there are no resources available. This status will usually be accompanied by the reason Priority, meaning your job is ready to go but there are no resources for it at this time. You may also see Resources, which means your job will have the requested resources reserved specifically for it (this status can mean your job will cause problems in the queue, so if you see it, either wait and submit your job later, or use advanced queueing to have it submitted only after a current job is finished; this is covered in the example .sh files). You may also see Dependency, which means your job is ready for it to go but is waiting for another of your jobs to finish running before it runs. 
- CG: your program is terminating. 

You may also check the status of partitions by running the squeue -p command, specifying either gpu or workq after the -p to see the queue for that partition. This can give you an idea of the current load on that partition and therefore how long you might expect it to take before your job is run. 

When your program runs, all output files will appear in the directory where your .sh file is. Unlike with terminal outputs, the output of a job is split into two files - one for outputs, and one for errors. This can make debugging easier if your job encounters an error when running. You can adjust the names of these files as desired. Note that you do not need to change the name on each running of the job, as Slurm will automatically append the job number to the end of the output and error files to distinguish them. 

The Slurm system can email you when specific events occur, such as when a queued job starts running or when it finishes. The commands for this are included in the example .sh files. 

// Updating Permissions for Project Files //

After you finish running a job, working in an interactive session, or otherwise add or edit files that are part of a project directory, make sure to update the permissions of those files so your teammates can access them. By default, only you will have read and write access to them. In order to grant others access, run the command chmod -R g+rw (directory name) in the parent directory of the directory containing the files you want to update the permissions for. Note that this could also be done with the chmod -R 0777 or similar commands, but this has previously resulted in a glitch wiping all permissions (even yours) to the files, and is so not recommended. The -R parameter ensures that permissions for all files in the folder are recursively updated. 






