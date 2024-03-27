# OptiVisT
# Helping Blind People Grasp: Evaluating a Tactile Bracelet for Remotely Guiding Grasping Movements

## Contributors
Piper Powell<sup>2</sup>, Florian Pätzold<sup>2</sup>, Milad Rouygari<sup>2</sup>, Marcin Furtak<sup>1,2</sup>, Silke Kärcher<sup>1</sup>, Peter König<sup>1,2</sup>  
<sup>1</sup>feelSpace GmbH, Albert-Einstein-Straße 1, 49076 Osnabrück  
<sup>2</sup>Institute of Cognitive Science, 49060 University Osnabrück

## Abstract
<p align="justify">
The problem of supporting visually impaired and blind people in meaningful interaction with objects is often neglected. To address this issue, we adapted a tactile belt for enhanced spatial navigation into a bracelet worn on the wrist that allows visually impaired people to grasp target objects. Participants' performance in locating and grasping target items when guided with the bracelet, which provides direction commands via vibrotactile signals, was compared to their performance when receiving auditory instructions. While participants were faster with the auditory commands, they also performed well with the bracelet, encouraging future development of this and similar systems.

## Repo & Use Guide
In this repo, you will find the code both for the blindfolded trials and for the early AI work. The code for the blindfolded trials is contained in the `flobox` folder, while the code for the AI paradigm is in the `aibox` folder. 

### FloBox (Blindfolded Trials)
Within the `flobox` folder, you will find the `Experiment` directory, which contains the code for running the blindfolded trials. To run this experiment, run the `tactile_exp_original.py` script. It will reference the `connect.py` script in the same folder to connect to the tactile bracelet. The `Data` folder in the `Experiment` directory contains the collected data (anonymous) for the blindfolded trials. The `Analysis` folder contains the analysis files for analyzing the collected data. In the `flobox` folder, you will also find the `old` directory, where old and no longer used files for the blindfolded trials are stored. Note that the file structure image in this directory is slightly out of date. 

### AiBox (AI Paradigm)
Within the `aibox` folder, you will find all code necessary for running the AI paradigm. The `old` directory also contains old files that are no longer used. The `Testing` and `Training` directories contain the files needed to train and test the network on a SLURM based high performance computing network (note that the actual training files that would be referenced in the training SLURM scripts are not present, as the networks were trained with the standard Ultralytics training code (`train.py`), obtainable from this [repo](https://github.com/ultralytics/yolov5)). It is not necessary to train the networks yourself to utilize our code here, as we provide the weights files for both networks being used. The `deep_sort_pytorch...` directory can be ignored; this was a directory for the early work on object tracking (the current software being solely for object recognition. The `models` and `utils` directories are copied from the Ultralytics software, as they are needed to run the YOLOv5 models used in our project (which are from the Ultralytics implementation, see the section below on installing Ultralytics). The `sound` directory contains the sound files which are called in the AI trials to announce the start and end of the experiment and the current object being grasped. 

Reference this file structure image for an overview of the scripts and folders needed to run the AI paradigm ![AIBox Paradigm](https://github.com/pippowell/OptiVisT/blob/main/aibox/file_guide_opti.png)

### Installing Ultralytics
For the networks to run, you need both the latest ultralytics package (which is actually for YOLOv8) and the separate package for YOLOv5. 

To set up an environment with both, run the following command to install the YOLOv8 package:
pip install ultralytics 

Then download the YOLOv5 repo from the following [link](https://github.com/ultralytics/yolov5/tree/master). You can either download as a ZIP, clone to a separate repo, etc.

Move the files from the downloaded repo to your current working repo (where the files you want to run are). Update any code references depending on where exactly in your current working repo the YOLOv5 files are. 

You do not need all the YOLOv5 files, just the following:
- the models folder 
- the utils folder
- export.py

If you clone this repo, YOLOv5 is already set up properly and you only need to separately install the YOLOv8 packages into your environment.
