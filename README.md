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

5) Install one missing msnoise dependency: 
  
   (skience23) $ ` conda install -c conda-forge pytables `
   
7) Test the environment using: 
   
   (skience23) $ ` obspy-runtests `
   
   (skience23) $ ` qopen-runtests `
   
   (skience23) $ ` msnoise utils test `
   
6) To eventually delete the environment again type (after the workshop, of course):

    ` conda env remove --name skience23 `

A quick explanation on conda environment setup with yaml files can also be found here: 
https://medium.com/@balance1150/how-to-build-a-conda-environment-through-a-yaml-file-db185acf5d22

If you have issues with the setup, please share the error messages on Mattermost -> Channel "Installation - Software Issues" !

## Software Setup 
On Tuesday we will use __Geopsy__. **Either** download the software on: https://www.geopsy.org/download.php. Select the platform you need (Windows, Linux, Max) and hit the green carton box icon to download Geopsy:  <img src="https://github.com/heinerigel/skience2023/blob/main/02_Tuesday/Afternoon/Figures/Geopsy%20download.png" width=30></img> 
__OR__ download Geopsy from the links below. No installation needed, just unpack the zip folder and place it in any document folder.

__Required versions__:
* Windows Geopsy 3.4.2: https://www.geopsy.org/download/archives/geopsypack-win64-3.4.2.zip 
* Linux Geopsy 3.4.2: https://www.geopsy.org/download/archives/geopsypack-src-3.4.2.tar.gz 
* Mac: Geopsy 3.3.6 (3.4 not available for download):  https://www.geopsy.org/download/archives/geopsypack-mac-bigsur-3.3.6.dmg 
