# TATOO
Python Topographic Analysis Tool Library for semi-automated Setup of High-resolution Water Balance Models

## 1. What the project does
TATOO offers Python 3.6 algorithms to automate the processes of water balance model parameter delineation. 
There are three packages (_tatoo_subcatchment.py_, _tatoo_raster.py_, and _tatoo_common.py_). The packages _tatoo_subcatchment.py and _tatoo_raster.py_ use functions from the package _tatoo_common.py_. Additionally, there are two example workflows (_tatoo_subcatchment_example.py_ and _tatoo_raster_example.py_), which use functions of the three function packages. The dependencies between these five files are summarized in this graph:

![here](https://user-images.githubusercontent.com/85393122/121018779-1c181580-c79f-11eb-98d2-eec9cc303ffc.png)

The example workflows apply all functionalities for the _Large Area Runoff Simulation Model (LARSIM)_ and explain all user parameters shortly. The model documentation can be found ![here](https://www.larsim.info/en/the-model/). Most functionalities are  applicable for many integrated hydrological models using same or similar parameters, only few are particularly for LARSIM (e.g., the model file creation functions).

## 2. The features and benefits of the project
The features depend on the model creation strategy (subcatchment or raster structure) as well as on the degree of detail the user wants to implement in the model. Basically, there are three sections to mention:
* raster- or subcatchment-based model elements and networks (including subcatchment size optimization), 
* their runoff generation, concentration and routing parameters, and 
* channel and foreland cross-section geometries (triple-trapezoid-profiles or level-area-perimeter-functions). 
While the first and the second are obligatory, the latter is voluntary. 

### Raster- or subcatchment-based model elements and networks
Every integrated hydrological model needs spatial model elements, which may be raster-, or subcatchment-based. For the creation of raster models, the user only determines the size of the grid cells and the functions will create the necessary model elements and the network dependencies indipendently. For subcatchment models, there is an optimizer included, which tries to fit the subcatchment sizes according to the user-selected size. The user has possibilities in both to correct the resulting elements and structures afterwards manually within GIS.

### Runoff generation, concentration and routing parameters
The functions presented here are constructed for a model using Kirpich (1940) as runoff concentration time estimation and a modified ![Williams (1969)](https://doi.org/10.13031/2013.38772) approach applying Gaukler-Manning-Strickler formula for channel routing within a triple (recommended) channel cross section profile. The latter is estimated standardally using channel estimation formula such as ![Allen et al. (1994)](https://doi.org/10.1111/j.1752-1688.1994.tb03321.x).

### Channel and foreland cross-section geometries
If the user would like more sophisticated data, than it is possible to either adapt automatically created cross sections (raster) or use averaged cross sections (subcatchments) as model input. While the inputs for the raster model are given then in a separate format, a triple-trapezoid profile is optimized for subcatchment models.

The functionalities are briefly described within the following detailled flow chart of TATOO functions and this publication:
Mitterer, J. A. (2021): _"TATOO – Python Topographic Analysis Tool Library for semi-automated Setup of High-resolution Water Balance Models"_. In: Environmental Modelling and Software, ## (##) pp. ###-###. Additionally, every function has its own detailled documentation within the .py files.

![tatoo flowchart](https://user-images.githubusercontent.com/85393122/121212460-ab93f600-c87d-11eb-8f0b-1c58f082f951.png)

## 3. How users can get started with the project
### System requirements
There are no specific system requirements except a running Python 3 environment. Anyhow, it could be helpful for trouble-shooting to know the test environment conditions: 
* processor: Intel(R) Xeon(R) CPU E5-1650 v4 @ 3.60 GHz
* memory/RAM: 32.0 GB RAM
* operating system: Microsoft Windows 10 Enterprise 2016 LTSB (version 10.0.14393 Build 14393)
* system type: x64-based system

### Necessary Python packages
The functions of TATOO are based on Python-3.6 and require a bunch of standard Python packages (see _getting_started.md_). Before using the functions of TATOO, the user has to install these packages successfully. Please make sure that they interact correctly with your ArcPy environment. We propose to use Anaconda as environment management portal. 
The tools need ESRI's Python package ArcPy in Python 3 version, which is included within ArcGIS Pro. The tools are indipendent of ArcGIS except for the ArcPy environment, although the user interactions were tested using ArcGIS Pro version 2.5. 

### Installation
Besides the packages, there is no special installation necessary. As described in the _example workflows_, the user only has to add the path, where the package files (_tatoo_subcatchment.py_, _tatoo_raster.py_, and _tatoo_common.py_) are located. The command is:
sys.path.append((r'PATH')) # replace PATH with your folder path

## 4. Where users can get help with the project
There is 
For the documentation of the functions 

The project was developed by Johannes Mitterer at the Technical University of Munich (TUM), ![Chair for Hydrology and River Basin Management](https://www.bgu.tum.de/en/hydrologie/home/) in 2021. Contact: johannes.mitterer@tum.de

## 5. Who maintains and contributes to the project
The project is recently published. Contributors:
* Johannes Mitterer (Technical University of Munich)

There are three possibilities to take part in this project:
1. apply the functions for LARSIM (or for other models)
2. raising issues about bugs (or missing functionalities) in GitHub 
3. apply to get contributor and modify or add functionalities
You are invited to join with any of the listed participation methods.

## Credentials
Input data provided for the show cases is from the following agencies:
* Bayerisches Landesamt für Umwelt (LfU, Bavarian Environmental Agency): Soil, land use, and impervious share data
* Bayerisches Landesamt für Digitalisierung, Breitband und Vermessung (LfDBV, Agency for Digitisation, High-Speed Internet and Surveying): Digital elevation model (1x1 m²)
The study to develop and publish this packages have been supported by the following institutions:
* Technische Universität München (TUM, Technical University of Munich), Prof. Dr.-Ing. Markus Disse
* Ludwig-Maximilians-Universität Munich (LMU), Department for Geography, Prof. Dr. Ralf Ludwig
* Bayerisches Landesamt für Umwelt (LfU, Bavarian Environmental Agency)
* Bayerisches Staatsministerium für Umwelt und Verbraucherschutz (StMUV, Bavarian State Ministry of the Environment and Consumer Protection
The snipped to convert a pandas.DataFrame to a numpy.StructuredArray for use in ArcPy.da.NumPyArrayToTable in the package _tatoo_common.py_ is from ![USGS developers](https://my.usgs.gov/confluence/display/cdi/pandas.DataFrame+to+ArcGIS+Table). Many thanks to them for this shortcut!
