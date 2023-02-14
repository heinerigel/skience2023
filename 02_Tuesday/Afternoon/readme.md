# Minionology: HVSR with seismic nodes
## Skience2023 practical on HVSR, node installation, applications, Geopsy, continuous data analysis

## Authors:
* Koen Van Noten ([@KoenVanNoten](https://github.com/KoenVanNoten))
* Thomas Lecocq ([@seismotom](https://github.com/ThomasLecocq)

## Introduction:
Three-component __seismic nodes__ are conquering the world these days as lightweight smart seismic sensors. This notebook is a simple introduction how to handle seismic node data and perform H/V spectral ratio analysis of ambient noise (mHVSR) recorded with seismic nodes. We'll show you some methods:
* how to perform mHVSR using the Geopsy software manually
* how to do the same exercise automatically 
* and we'll open the discussion towards more automatic solutions given by the Skience2013 crowd. 

SmartSolo 3D Nodes (https://smartsolo.com/cp.php?id=3) are easy to deploy, have long battery life (2-4 weeks), are modular to easily replace battery and are fastly charged. The picture below shows the modular design of the IGU-16HR 3C series where 3C sensors are installed on a standard battery pack (usually used with the 1C nodes). The tripod feet allow them to be used in urban settings. As they resemble to the Minions, we batised _data analysis with nodes_ as __Minionology__. We further refer to Zeckra et al. (submitted) to a technical introduction on the performance of the IGU-16HR 3C series. 

## References:
* Van Noten, K., Devleeschouwer, X., Goffin, C., Meyvis, B., Molron, J., Debacker, T.N. & Lecocq, T. 2022. Brussels’ bedrock paleorelief from borehole-controlled powerlaws linking polarised H/V resonance frequencies and sediment thickness. _Journal of Seismology_ 26, 35-55. DOI: https://doi.org/10.1007/s10950-021-10039-8 pdf: https://publi2-as.oma.be/record/5626/files/2022_VanNotenetal_HVSR_Powerlaw_Brussels.pdf 
* Van Noten, K, Lecocq, Buddha Power, B. (2020). HVSR to Virtual Borehole (1.0). Zenodo. https://doi.org/10.5281/zenodo.4276310
* Zeckra, M., Van Noten, K., Lecocq, T. Submitted. Sensitivity, Accuracy and Limits of the Lightweight Three-Component SmartSolo Geophone Sensor (5 Hz) for Seismological Applications. _Seismica_. Preprint on: https://doi.org/10.31223/X5F073

<img src="Figures/Minions Seismology.be.jpg" width=600></img>
