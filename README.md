# skience2023

Welcome!

This is the github Project for the 2023 Skience Winter School on Environmental Seismology. Training material for the lectures and practices is provided here.

https://www.skience.de/2023


## Setup of Conda Environment for Skience23

[Miniconda/Anaconda3](https://docs.conda.io/en/latest/miniconda.html) has to be installed for this to work!

1) Download the [__skience23.yaml__](https://raw.githubusercontent.com/heinerigel/skience2023/main/skience23.yaml) file from this repository! This file specifies all required packages for the environment. Warning: make sure the content of the file is correct and not raw html!

2) Using the Anaconda Prompt (or your favorite console if it works): use conda to create an environment: 
  
   ` conda env create -f <path_to_yaml_file> `

3) After this terminated successfully, you should be able to list the environment: 
   
   ` conda env list `
   
   and to activate it using: 
   
   ` conda activate skience23 `

   When activated, your command line should show:
   
   ` (skience23) $ `  

4) Test the environment using: 
   
   (skience23) $ ` obspy-runtests `
   
   (skience23) $ ` qopen-runtests `
   
   (skience23) $ ` msnoise utils test `
   
5) To eventually delete the environment again type: 

    ` conda env remove -name skience23 `

A quick explanation on conda environment setup with yaml files can also be found here: 
https://medium.com/@balance1150/how-to-build-a-conda-environment-through-a-yaml-file-db185acf5d22
