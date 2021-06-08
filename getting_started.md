## 1. Installation instructions

### System requirements
The functions of TATOO are based on Python-3.6 and require a bunch of standard Python packages:
* os
* sys
* copy
* datetime
* lmfit 1.0.1
* arcpy from ArcGisPro 2.5
* numpy (incl. matlib and lib.recfunctions) 1.16.6
* pandas 1.0.5
* matplotlib (pyplot and mpl_toolkits.axes_grid1.inset_locator) 3.1.1
* osgeo.gdal 2.3.3
* scipy (interpolate.griddata, integrate.cumtrapz, signal.find_peaks) 1.5.0
Before using the functions of TATOO, the user has install the named packages correctly. Please make sure that the interact correctly with your ArcPy environment. We propose to use Anaconda as environment management portal. 

### ArcPy requirements
The tools need ESRI's Python package ArcPy in Python 3 language, which is included within ArcGIS Pro. The tools are indipendent of ArcGIS except for the ArcPy environment, although the user interactions were tested using ArcGIS Pro version 2.5. 

### Installation
Besides the packages, there is no special installation necessary. As in the _example workflows_, the user only has to add the path, where the package files (_tatoo_subcatchment.py_, _tatoo_raster.py_, and _tatoo_common.py_) are located. The command is:
sys.path.append((r'PATH')) # replace PATH with your folder path

## 2. Completing essential tasks

### First Application
For the first start with TATOO, we recommend to build on the existing workflows (_tatoo_subcatchment_example.py_ and _tatoo_raster_example.py_), as there all parameters and potential functions are used. Then the user can have a look into the output data structure and decide if modified (or even new) functions are necessary to fit the individual model requirements. Both examples have the same structure:
* **header**: description of functionalities, authors and rights
* **import**: used Python packages
* **control**: paths and parameters
* **intermediates**: paths and names used throughout the project
* **model elements and networks**: functions concerning model structure, runoff concentration and routing parameters
* **runoff generation**: impervious area, land use, and soil parameters
* **detailled cross section information** (raster package only)
Users may now apply the functions step-by-step or adapt them dependent on their needs. A complete run of the example workflow skripts will result in a complete model structure for LARSIM without any user changes.

### Interaction with ArcGIS Pro
According to the workflows, we recommend to set up two personal geodatabases (GDB) as working folders for inputs and outputs. The inputs shall contain five files:
* high-resolution digital elevation model (obligatory)
* land use polygon feature class containing all identifier numbers for the project (obligatory)
* soil polygon feature class containing all necessary parameters (obligatory)
* polyline feature class of flow network (obligatory can be empty as well)
* impervious share polygon feature class with float values (optional)
The functions generally summarize parameters in pandas.DataFrames and export these to ArcPy.FeatureClasses, ArcPy.Tables or ArcPy.Rasters if required. User can manually change these ONLY according to the outlines within the workflow files if they are used as inputs for other functions.

## 3. Customizing and configuring
--> documentation material
