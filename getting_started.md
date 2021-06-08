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

### Paths
#### Geographical input data

##### path_gdb_in
path of input GDB, where geographical input data is expected _(e.g., r'PATH\input.gdb' + '\\')_

##### path_dem_hr
path of high-resolution DEM with burned river network _(e.g., path_gdb_in + 'dem_hr')_

##### path_fnw
path of flow network polyline feature class _(e.g., path_gdb_in + 'fnw')_

##### path_lu
path of polygon feature class of land use and sealing percentage _(e.g., path_gdb_in + 'landuse')_

##### path_soil
path of polygon feature class of soil data _(e.g., path_gdb_in + 'soil')_

#### Geographical output data
##### path_gdb_out
geodatabase where outputs are stored _(e.g., r'PATH\output.gdb' + '\\'  
##### name_ws_s
model domain polygon feature class _(e.g., 'ws_s')_
##### name_tgb_s
model element polygon feature class _(e.g., 'tgb_s')_
##### name_tgb_par_p
model element point feature class incl. raster parameters _(e.g., 'tgb_par_p')_
##### name_tgb_par_tab
model element table with output file parameters _(e.g., 'tgb_par_tab')_
##### name_hru_c
polygon feature class of HRUs _(e.g., 'hru_c')_
##### name_utgb_par_tab
HRU table with output file parameters _(e.g., 'utgb_par_tab')_
##### name_csl
cross section polyline feature class _(e.g., 'csl')_
##### name_bcwsl
bankful channel water surface line _(e.g., 'bcwsl')_
##### name_profile_par
parameter table for bankful discharge _(e.g., 'profile_par')_
##### name_ch_fit_coords
coordinate table for bankful discharge line _(e.g., 'ch_fit_coords')_
#### Output path and file names
path_files_out     = r'PATH\model_files' + '\\' folder where files are stored
name_tgbdat        = 'tgb.dat'                  LARSIM file for model element parameters
name_utgbdat       = 'utgb.dat'                 LARSIM file for HRU element parameters
name_profdat       = 'profile.dat'              LARSIM file for detailled cross section information (only raster models)

### Parameters with recommended or examplary values
#### parameters for high-resolution elevation model processing
model_cellsize     = 200                        model cell size in [m]
burning_h          =   5                        deepening value at flow network in [m]
#### parameters for runoff concentration processing
def_sl_min         =   0.0001                   minimum slope (LARSIM internal convention: max. 4 decimals, but ~= 0) in [mZ/mL]
def_fl_strct_mism  =   2                        flow length for structural mismatch and negative transition deviations in [m]. LARSIM: 1 = 'no routing'!
def_fl_upper_lim   = 300                        upper threshold for realistic flow length in [m]
def_sl_excl_quant  =   0.999                    upper threshold for realistic slope as quantile in -]
def_zmin_rout_fac  =   0.5                      percentage of stream's low and high point to define runoff concentration z-difference [-]
def_zmax_fac       =   1.0                      elevation percentage of cell to define runoff concentration z-difference [-]
#### define parameters for channel estimation
hq_ch              =   8.0                      HQ2 (or MHQ summer) as float in [m³/s]
hq_ch_a            = 175.5                      catchment area as float in [km²]
ser_q_in_corr      = pd.Series(np.array([]), index=[], name='q_in')   inflows not represented in the model structure (ascending order of inflow cell IDs [-] as list of integers (index) and HQ2 (or MHQ) of inflow cells [m³/s] as list of floats (data), (e.g., pd.Series(np.array([34.5, 3.2]), index=[346, 687], name='q_in'))
ch_est_method      = 'combined'                 string defining channel estimation function, can be 'Allen', 'Krauter' or 'combined'
def_bx             =   0                        foreland edge default width in [m] (0 = foreland not existing)
def_bbx_fac        =   1                        foreland default factor (bbl = bm * def_bbx_fac, bbr = bm * def_bbx_fac) [-]
def_bnm            =   1.5                      channel slope in [mL/mZ] (1.5 = 67% slope)
def_bnx            = 100                        foreland slope in [mL/mZ] (100 = 1% slope, ~ even)
def_bnvrx          =   4                        foreland boarder slope in [mL/mZ] (4 = 25% slope)
def_skm            =  30                        river channel roughness values in [m1/3s-1] (30 = natural, vegetated river bank)
def_skx            =  20                        foreland rouighness value in [m1/3s-1] (20 = uneven vegetated )
#### parameters for hydrological response unit handling (LARSIM-specific)
ctrl_opt_infdyn    = True     control operator to activate dynamic infiltration routine parametrization (False: no parametrization, True: parametrization)
ctrl_opt_impperc   = True     control operator to activate sealing parametrization in utgb.dat (False: no parametrization, True: parametrization)
ctrl_opt_capr      = False    control operator to activate capillary rise parametrization in utgb.dat (False: no parametrization, True: parametrization)
ctrl_opt_siltup    = True     control operator to activate silting-up parametrization in utgb.dat (False: no parametrization, True: parametrization)
ctrl_hru_aggr      = True     control operator to define HRU aggregation (False: no aggregation, True: aggregate)
def_amin_utgb_del  = 10**-8   area threshold below which HRUs are deleted in [m²]
#### input field names of GIS inputs for HRU calculation
f_lu_id            = 'landuse_id'    landuse class ID numbers (in path_lu)
f_impperc          = 'imp_perc'      user-defined impervious percent value (in path_lu)
#### assignment table for land use classes and parameters
arr_lu_mp          = np.array([[landuse_id, landuse_name, MPla, MPdi],
                               [landuse_id, landuse_name, MPla, MPdi]])
                                     with land use class IDs (landuse_id), names (landuse_name) and parameters for macropore density (MPla) and length (MPla)
lu_id_imp          =   3             land use class ID for impervious land use
lu_id_water        =  16             land use class ID for water

#### detailled cross section delineation
general parameters
ctrl_rout_cs_file      = True        (de-)activate creating cross section profile parameter file
ctrl_show_plots        = False       (de-)activate showen windows of plots (attention: every cross section creates one new window!)
ctrl_save_plots        = True        (de-)activate storing of plots on hard drive
path_plots_out         = r'PATH' + '\\' path, where figures shall be saved

parameters for cross section identification
def_cs_dist            = 200         distance between transects in [m]
def_river_a_in         =   1         minimum catchment area defining a river in [km²]
def_cs_search_rad      =   5         search radius for transect identification in [m]
def_cs_wmax_eval       = 600         length of transects (incl. both sides of river) in [m]

parameters for cross section water level - discharge curve determination
def_cs_intp_buf_dist   =   1         cross section intersection points' buffer distance in [m]
def_cs_hmax_eval       =  10         maximum height of cross section evaluation in [m]
def_lam_hres           =   0.1       vertical spacing between evaluation lamellae in [m]
def_ch_wmax_eval       =  40         maximum width of channel evaluation in [m]
def_ch_hmin_eval       =   0.1       minimum height of channel evaluation in [m]
def_ch_hmax_eval       =   3.0       maximum height of channel evaluation in [m]
def_ch_hmin            =   0.2       minimum channel depth threshold for channel identification in [m], must be >= 0.2
def_ch_vmin            =   0.5       minimum reasonable flow velocity in [m/s]
def_ch_vmax            =   3.0       maximum reasonable flow velocity in [m/s]
def_chbank_slmin       =   0.1       minimum channel bank slope threshold def_lam_hres/dL for channel identification [-]
def_ch_w               =   0.5       artificial channel width, added to continuiously descending cross sections in [m]
def_ch_h               =   0.5       artificial channel depth, added to continuiously descending cross sections in [m]
def_ch_wres            =   0.05      horizontal resolution of interpolated points within channel in [m]

#### parameters for all model files
catch_name             = ''          catchment name (one line string, print in header of file)
src_geodata            = ''          source of used geoinformation (one line string, print in header of file)
#### parameters for model element parameter file tgb.dat
tgbdat_comment         = ''          comment (one line string, print in header of file)
hcs_epsg               = 25833       horizontal coordinate system EPDG number [-] (e.g., 25833)
vcs_unit               = 'm ue. NN'  vertical coordinate system units
def_tgb_nodata_val     =    -1       no data value of the tgb.dat file (max. 3 characters) [-]
#### parameters for model element parameter file utgb.dat
utgbdat_comment        = ''          comment (one line string, print in header of file)
def_utgb_nodata_val    =    -1       no data value of the utgb.dat file (max. 3 characters) [-]
#### parameters for model element parameter file profile.dat
profdat_comment        = ''          comment (one line string, print in header of file)
def_profdat_decmax     =      2      decimal numbers allowed in the file profile.dat [-]
def_profdat_nodata_val =     -1      no data value of the profile.dat file (max. 3 characters) [-]
def_profdat_exit_val   = 999999      value terminating cross section data block [-]
#### Command line output
print_out              = True        swith command line output on (True) or off (False)

### Raster workflow

### Subcatchment workflow
