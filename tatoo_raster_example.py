# -*- coding: utf-8 -*-
"""
This module contains an calculation example to produce the spatial files
of a LARSIM raster model (tgb.dat, utgb.dat, profile.dat).
It uses functions of the libraries 'TATOO core' and 'TATOO raster'.

Author: Johannes Mitterer
        Chair for Hydrology and River Basin Management
        Technical University of Munich

Requires the following ArcGIS licenses:
    - Conversion Toolbox
    - Spatial Analyst
    
System requirements:
    - Processor:  no special requirements
                  tested with Intel(R) Xeon(R) CPU E5-1650 v4 @ 3.60 GHz
    - Memory/RAM: depending on the size of the DEM to be processed
                  tested with 32,0 GB RAM
    - Python IDE for Python 3
    - ArcGIS Pro 2.5

Version: v1.0.0, 2021-05-02
"""
__author__     = 'Mitterer Johannes'
__copyright__  = 'Copyright 2021'
__credits__    = ''
__license__    = 'CC BY-NC-ND 3.0 DE'
__version__    = '1.0.0'
__maintainer__ = 'Johannes Mitterer'
__email__      = 'johannes.mitterer@tum.de'
__status__     = 'Production'

# %% import section
import sys
import arcpy
import numpy as np
import pandas as pd
sys.path.append((r'a:\tatoo')) # set path for packages
import tatoo_raster as tr
import tatoo_common as tc

# %% control section
# define input layers
# geodatabase where inputs are stored
path_gdb_in        = r'a:\tatoo\raster_example\input.gdb' + '\\'
path_dem_hr        = path_gdb_in + 'dem_hr'  # high resolution DEM (with burned river network)
path_fnw           = path_gdb_in + 'fnw'     # flow network polyline feature class
path_lu            = path_gdb_in + 'landuse' # polygon feature class of land use and sealing percentage
path_soil          = path_gdb_in + 'soil'    # polygon feature class of soil data
# define output layers
# geodatabase where outputs are stored
path_gdb_out       = r'a:\tatoo\raster_example\output.gdb' + '\\'
name_ws_s          = 'ws_s'          # model domain polygon feature class
name_tgb_s         = 'tgb_s'         # model element polygon feature class
name_tgb_par_p     = 'tgb_par_p'     # model element point feature class incl. raster parameters
name_tgb_par_tab   = 'tgb_par_tab'   # model element table with output file parameters
name_hru_c         = 'hru_c'         # polygon feature class of HRUs
name_utgb_par_tab  = 'utgb_par_tab'  # HRU table with output file parameters
name_csl           = 'csl'           # cross section polyline feature class
name_bcwsl         = 'bcwsl'         # bankful channel water surface line
name_profile_par   = 'profile_par'   # parameter table for bankful discharge
name_ch_fit_coords = 'ch_fit_coords' # coordinate table for bankful discharge line
# define output path and file names
path_files_out     = r'a:\tatoo\raster_example\model_files' + '\\'
name_tgbdat        = 'tgb.dat'
name_utgbdat       = 'utgb.dat'
name_profdat       = 'profile.dat'

# define parameters for high-resolution elevation model processing
# model cell size in meters
model_cellsize    = 200      # [m]
# deepening value at flow network in meters
burning_h         =   5      # [m]

# define parameters for runoff concentration processing
# define minimum slope (LARSIM internal convention: max. 4 decimals, but ~= 0)
def_sl_min        =   0.0001 # [mZ/mL]
# set default flow length for structural mismatch and negative transition deviations
# attention: 1 is interpreted by LARSIM as 'no routing'!
def_fl_strct_mism =   2      # [m]
# set upper threshold for realistic flow length
def_fl_upper_lim  = 300      # [m]
# set upper threshold for realistic slope as quantile
def_sl_excl_quant =   0.999  # [-]
# define HUT for cells with stream as 50% of stream's low and high point
def_zmin_rout_fac =   0.5    # [-]
# define HOT as highest point of cell (as defined by Kirpich 1940)
def_zmax_fac      =   1      # [-]

# define parameters for channel estimation
# define parameters for geomorphology
hq_ch             =   8.0    # [m³/s] HQ2 (or MHQ summer) as float
hq_ch_a           = 175.5    # [km²] catchment area as float
# ascending order of inflow cell IDs [-] as list of integers (index) and
# HQ2 (or MHQ) of inflow cells [m³/s] as list of floats (data)
# (e.g., pd.Series(np.array([34.5, 3.2]), index=[346, 687], name='q_in'))
ser_q_in_corr     = pd.Series(np.array([]), index=[], name='q_in')
# string defining channel estimation function
# may be 'Allen', 'Krauter' or 'combined'
ch_est_method     = 'combined'
# foreland edge default width
def_bx            =   0      # [m] (0 = not existing)
# foreland default factor (bbl = bm * def_bbx_fac, bbr = bm * def_bbx_fac)
def_bbx_fac       =   1      # [-]
# channel slope
def_bnm           =   1.5    # [mL/mZ] (1.5 = 67% slope)
# foreland slope
def_bnx           = 100      # [mL/mZ] (100 = 1% slope, ~ even)
# foreland boarder slope
def_bnvrx         =   4      # [mL/mZ] (4 = 25% slope)
# rouighness values
def_skm           =  30      # [m1/3s-1] (natural river channel, vegetated river bank)
def_skx           =  20      # [m1/3s-1] (uneven vegetated foreland)

# define parameters for hydrological response unit handling
# control operator to activate dynamic infiltration routine parametrization
ctrl_opt_infdyn   = True     # False: no parametrization, True: parametrization
# control operator to activate sealing parametrization in utgb.dat
ctrl_opt_impperc  = True     # False: no parametrization, True: parametrization
# control operator to activate capillary rise parametrization in utgb.dat
ctrl_opt_capr     = False    # False: no parametrization, True: parametrization
# control operator to activate silting-up parametrization in utgb.dat
ctrl_opt_siltup   = True     # False: no parametrization, True: parametrization
# control operator to define HRU aggregation
ctrl_hru_aggr     = True     # False: no aggregation, True: aggregate
# define area threshold below which HRUs are deleted
def_amin_utgb_del =  10**-8  # [m²]

# define input field names of GIS inputs for HRU calculation
f_lu_id           = 'landuse_id' # landuse class ID numbers (in path_lu)
f_impperc         = 'imp_perc'   # user-defined impervious percent value (in path_lu)
# assignment table for land use classes and parameters in the following order:
# land use class IDs (landuse_id), names (landuse_name) and parameters for
# macropore density (MPla) and length (MPla)
#                     [landuse_id, landuse_name, MPla, MPdi]
arr_lu_mp = np.array([[ 1, 'settlement'          ,   0,   0],
                      [ 2, 'space in settlement' ,   0,   0],
                      [ 3, 'impervious land'     ,   0,   0],
                      [ 4, 'cropland'            ,  75,  30],
                      [ 5, 'vineyard'            ,  75,  50],
                      [ 6, 'fruit growing'       , 100,  50],
                      [ 7, 'fallow land'         , 100,  80],
                      [ 8, 'unvegetated soil'    ,  75,  30],
                      [ 9, 'intensive pasture'   , 100,  80],
                      [10, 'wetland'             , 100,  30],
                      [11, 'extensive pasture'   , 100,  80],
                      [12, 'grassland with trees', 125,  65],
                      [13, 'conifer forest'      , 150,  30],
                      [14, 'broadleaf forest'    , 150,  50],
                      [15, 'mixed forest'        , 150,  50],
                      [16, 'water'               ,   0,   0]])
df_lu_mp = pd.DataFrame(arr_lu_mp[:,1:], index=arr_lu_mp[:,0].astype(np.int),
                        columns=(['landuse_name', 'MPdi', 'MPla'])).astype(
                                {'landuse_name': 'string',
                                 'MPdi': np.int, 'MPla': np.int})
# define specific land use class IDs
lu_id_imp            =   3    # impervious land use class
lu_id_water          =  16    # water land use class

# (de-)activate creating cross section profile parameter file
ctrl_rout_cs_file    = True
# define parameters for cross section identification
# distance between transects
def_cs_dist          = 200    # [m]
# minimum catchment area defining a river
def_river_a_in       =   1    # [km²]
# search radius for transect identification
def_cs_search_rad    =   5    # [m]
# length of transects (incl. both sides of river)
def_cs_wmax_eval     = 600    # [m]
# define parameters for cross section water level - discharge curve determination
# cross section intersection points' buffer distance
def_cs_intp_buf_dist =   1    # [m]
# maximum height of cross section evaluation
def_cs_hmax_eval     =  10    # [m]
# vertical spacing between evaluation lamellae
def_lam_hres         =   0.1  # [m]
# maximum width of channel evaluation in [m]
def_ch_wmax_eval     =  40    # [m]
# minimum and maximum height of channel evaluation in [m]
def_ch_hmin_eval     =   0.1  # [m]
def_ch_hmax_eval     =   3.0  # [m]
# minimum channel depth threshold for channel identification, must be >= 0.2
def_ch_hmin          =   0.2  # [m]
# minimum reasonable flow velocity
def_ch_vmin          =   0.5  # [m/s]
# maximum reasonable flow velocity
def_ch_vmax          =   3.0  # [m/s]
# minimum channel bank slope threshold def_lam_hres/dL for channel identification
def_chbank_slmin     =   0.1  # [-]
# artificial channel width and depth, added to continuiously descending cross sections
def_ch_w             =   0.5  # [m]
def_ch_h             =   0.5  # [m]
# horizontal resolution of interpolated points within channel
def_ch_wres          =   0.05 # [m]
# (de-)activate plot output
ctrl_show_plots      = False  # show plots?
ctrl_save_plots      = True   # save plots?
# define path, where figures shall be saved
path_plots_out       = path_files_out + r'fig' + '\\'

# define parameters for all model files
# catchment name (one line string, print in header of file)
catch_name             = 'Alteneich'
# source of used geoinformation (one line string, print in header of file)
src_geodata            = 'Bavarian Environmental Agency (LfU)'

# define parameters for model element parameter file tgb.dat
# comment (one line string, print in header of file)
tgbdat_comment         = 'Oberstimm / Brautlach (13249007) HQ2 (Alteneich: 14.8 km²)'
# horizontal coordinate system EPDG number
hcs_epsg               = 25833 # [-]
# vertical coordinate system units
vcs_unit               = 'm ue. NN'
# no data value of the tgb.dat file (max. 3 characters)
def_tgb_nodata_val     =    -1 # [-]

# define parameters for model element parameter file utgb.dat
# comment (one line string, print in header of file)
utgbdat_comment        = 'Oberstimm / Brautlach (13249007) HQ2 (Alteneich: 14.8 km²)'
# no data value of the utgb.dat file (max. 3 characters)
def_utgb_nodata_val    =    -1 # [-]

# define parameters for model element parameter file profile.dat
# comment (one line string, print in header of file)
profdat_comment        = ''
# decimal numbers allowed in the file profile.dat
def_profdat_decmax     =      2 # [-]
# no data value of the profile.dat file (max. 3 characters)
def_profdat_nodata_val =     -1 # [-]
# value terminating cross section data block
def_profdat_exit_val   = 999999 # [-]

# swith command line output on (True) or off (False)
print_out = True

# %% calculation section
# set intermediate feature class and raster names
name_dem_mr_f        = 'dem_mr_f'   # filled model-resolution elevation model
name_pp              = 'pp'         # model watershed pour point feature class
name_fd_mr           = 'fd_mr'      # model-resolution flow direction raster
name_fd_mr_corr      = 'fd_mr_corr' # corrected model-resolution flow direction raster
name_fa_mr           = 'fa_mr'      # model-resolution flow accumulation raster
name_fa_hr           = 'fa_hr'      # high-resolution flow accumulation raster
name_fd_p_corr       = 'fd_p_corr'  # flow-direction correction point feature class
name_fl_mr           = 'fl_mr'      # model-resolution flow length raster
name_tgb_p           = 'tgb_p'      # point feature class representing model elements' center
name_mnw             = 'mnw'        # model structure polyline feature class
name_no_fnw_fl       = 'no_fnw_fl'  # point feature class preventing flow length correction
name_dem_max_mr      = 'dem_max_mr' # model resolution aggregated max DEM values
name_dem_min_mr      = 'dem_min_mr' # model resolution aggregated min DEM values
name_fl_fnw_mr       = 'fl_fnw_mr'  # model resolution flow length calculated from flow network
name_dem_hr_ws       = 'dem_hr_ws'  # high resolution DEM raster clipped to model domain
name_fnw_o           = 'fnw_o'      # reference flow network polyline feature class
# set field names
field_pp_ws          = 'ModelWatershed' # model watershed definition field
field_dem_max_mr     = 'dem_max_mr'     # maximum elevation within model elements
field_dem_min_mr     = 'dem_min_mr'     # minimum elevation within model elements
field_fl_fnw_mean_mr = 'fl_fnw_mean_mr' # mean high-resolution flow length within model elements
field_fl_mr          = 'fl_mr'          # model-resolution flow length withing model elements

# %% tgb.dat: model structure file including runoff concentration and routing parameters
# aggregate digital elevation raster, fill, and burn flow network
tr.preprocess_dem(path_dem_hr, path_fnw, model_cellsize, burning_h,
               path_gdb_out, name_dem_mr_f=name_dem_mr_f,
               print_out=print_out)
# create pour point feature class
tc.create_pourpoint(path_fnw, path_gdb_out, name_pp=name_pp,
                    field_pp=field_pp_ws, print_out=print_out)

# --> manually create pour point features in ArcGIS
#     1. load in GIS software:
#        a. the filled digital elevation model (name_dem_mr_f) 
#        b. the model watershed pour point feature class (name_pp) 
#     2. create pour point features defining outflow points of
#        a. the watershed defining the model domain
#        b. upstream watersheds, which shall be excluded from the model domain
#     3. define integer numbers in the variable 'ModelWatershed'
#        a. using positive integers for watersheds within the model domain (i.e., 2)
#        b. using negative integers for watersheds, which shall be excluded (i.e., -1)

# calculate initial model watershed
path_dem_mr_f = path_gdb_out + name_dem_mr_f
path_pp       = path_gdb_out + name_pp
tr.calc_watershed(path_dem_mr_f, path_pp,
               path_gdb_out, name_fd_mr=name_fd_mr, name_fd_mr_corr=name_fd_mr_corr, 
               name_fa_mr=name_fa_mr, name_ws_s=name_ws_s, 
               initial=True, name_fd_p_corr=name_fd_p_corr, path_fd_p_corr='', 
               print_out=print_out)

# --> manually correct model watershed creating flow direction correction points
#     1. load in GIS software:
#        a. the model watershed polygon feature class (name_ws_s) 
#        b. the flow direction correction point feature class (path_fd_p_corr) 
#     2. create point features defining cells whose flow direction shall be manipulated
#        a. cells, which are outside the model domain, but shall be within
#        b. cells, which are included in the model domain, but shall be excluded
#     3. define integer numbers representing the D8 flow direction in the variable 'D8'
#        (E: 1, SE: 2, S: 4, SW: 8, W: 16, NW: 32, N: 64, NE: 128)

# recalculate model watershed with corrected flow direction values
path_fd_p_corr  = path_gdb_out + name_fd_p_corr
tr.calc_watershed(path_dem_mr_f, path_pp,
               path_gdb_out, name_fd_mr=name_fd_mr, name_fd_mr_corr=name_fd_mr_corr,
               name_fa_mr=name_fa_mr, name_ws_s=name_ws_s, 
               initial=False, name_fd_p_corr='', path_fd_p_corr=path_fd_p_corr, 
               print_out=print_out)
# calculate model network
path_ws_s       = path_gdb_out + name_ws_s
path_fd_mr_corr = path_gdb_out + name_fd_mr_corr
path_fa_mr      = path_gdb_out + name_fa_mr
_, _ = tr.calc_model_network(path_ws_s, path_fd_mr_corr, path_fa_mr, 
                   path_gdb_out, path_files_out, name_fl_mr=name_fl_mr,
                   name_tgb_p=name_tgb_p, name_mnw=name_mnw, 
                   print_out=print_out)

# --> manually correct model network creating flow direction correction points
#     1. load in GIS software:
#        a. the model network polyline feature class (name_mnw) 
#        b. the flow direction correction point feature class (path_fd_p_corr) 
#     2. create point features defining cells whose flow direction shall be manipulated
#     3. define integer numbers representing the D8 flow direction in the variable 'D8'
#        (E: 1, SE: 2, S: 4, SW: 8, W: 16, NW: 32, N: 64, NE: 128)
#     Attention: do not produce intersecting (crossing) flow network connections!

# recalculate model watershed and network
tr.calc_watershed(path_dem_mr_f, path_pp,
               path_gdb_out, name_fd_mr=name_fd_mr, name_fd_mr_corr=name_fd_mr_corr,
               name_fa_mr=name_fa_mr, name_ws_s=name_ws_s, 
               initial=False, name_fd_p_corr='', path_fd_p_corr=path_fd_p_corr, 
               print_out=print_out)
df_data_tgb_p, df_j_up = tr.calc_model_network(
        path_ws_s, path_fd_mr_corr, path_fa_mr, 
        path_gdb_out, path_files_out=path_files_out, name_fl_mr=name_fl_mr,
        name_tgb_p=name_tgb_p, name_mnw=name_mnw, 
        print_out=print_out)
# create polygon feature class representing the model elements
path_tgb_p     = path_gdb_out + name_tgb_p
path_sn_raster = path_fd_mr_corr
tr.create_element_polyg(path_tgb_p, path_sn_raster, path_gdb_out,
                     name_tgb_s=name_tgb_s, print_out=print_out)

# preprocess rasters for routing parameter calculation
tr.prepr_routing_rasters(path_dem_hr, path_fnw, path_ws_s, path_fa_mr, 
                      path_gdb_out, name_fa_hr=name_fa_hr, 
                      name_dem_hr_ws=name_dem_hr_ws, name_fl_fnw_mr=name_fl_fnw_mr,
                      name_dem_max_mr=name_dem_max_mr, name_dem_min_mr=name_dem_min_mr,
                      initial=True, print_out=print_out)
# create point feature class to indicate model cells without flow network flow length calc.
sr_obj = arcpy.Describe(path_fnw).spatialReference
tr.create_fl_ind_point(sr_obj, path_gdb_out, name_no_fnw_fl=name_no_fnw_fl,
                       print_out=print_out)

# --> manually create points at locations, where no flow network
#     flow length shall be calculated
#     1. load in GIS software:
#        a. the model network polyline feature class (name_mnw) 
#        b. the point feature class identifying elements witn no
#           flow network flow length calculation (name_no_fnw_fl) 
#        c. the raster representing the flow network flow length (name_fl_fnw_mr) 
#     2. create point features defining cells where no flow network
#        flow length shall be calculated

# summarize GIS data for runoff concentration and routing parameter calculation
path_dem_max_mr = path_gdb_out + name_dem_max_mr
path_dem_min_mr = path_gdb_out + name_dem_min_mr
path_fl_mr      = path_gdb_out + name_fl_mr
path_fl_fnw_mr  = path_gdb_out + name_fl_fnw_mr
path_no_fnw_fl  = path_gdb_out + name_no_fnw_fl
tr.summar_gisdata_for_roandrout(
    path_tgb_p, path_dem_max_mr, path_dem_min_mr, path_fl_mr, path_fl_fnw_mr, path_no_fnw_fl, 
    path_gdb_out, name_tgb_par_p=name_tgb_par_p,
    field_dem_max_mr=field_dem_max_mr, field_dem_min_mr=field_dem_min_mr,
    field_fl_fnw_mean_mr=field_fl_fnw_mean_mr, field_fl_mr=field_fl_mr,
    print_out=print_out)
# calculate parameters for tgb.dat
q_spec_ch = hq_ch / hq_ch_a # calculate channel forming specific discharge 
path_tgb_par_p = path_gdb_out + name_tgb_par_p
df_data_tgbdat, ser_tgb_down_nd, ser_ft, ser_area_outfl, ser_ch_form_q = \
    tr.calc_roandrout_params(
        model_cellsize, q_spec_ch, path_tgb_par_p,
        field_dem_max_mr=field_dem_max_mr, field_dem_min_mr=field_dem_min_mr, 
        field_fl_mr=field_fl_mr, field_fl_fnw_mean_mr=field_fl_fnw_mean_mr,
        def_fl_upper_lim=def_fl_upper_lim, def_fl_strct_mism=def_fl_strct_mism,
        def_sl_min=def_sl_min, def_sl_excl_quant=def_sl_excl_quant,
        def_zmin_rout_fac=def_zmin_rout_fac, def_zmax_fac=def_zmax_fac,
        ser_q_in_corr=ser_q_in_corr, ch_est_method=ch_est_method,
        def_bx=def_bx, def_bbx_fac=def_bbx_fac, def_bnm=def_bnm, def_bnx=def_bnx,
        def_bnvrx=def_bnvrx, def_skm=def_skm, def_skx=def_skx, 
        print_out=print_out)
# write tgb.dat file
path_tgbdat = path_files_out + name_tgbdat
tc.write_tgbdat(df_data_tgbdat, path_tgbdat, def_tgb_nodata_val=def_tgb_nodata_val, 
             hcs_epsg=hcs_epsg, vcs_unit=vcs_unit,
             src_geodata=src_geodata, catch_name=catch_name, 
             comment=tgbdat_comment, print_out=print_out)
# write model element data to ArcGIS table and join to subcatchment polygons
tc.df_to_table(df_data_tgbdat, path_gdb_out, name_tgb_par_tab)
# join fields to polygon feature class of model elements
path_tgb_s = path_gdb_out + name_tgb_s
path_tgb_par_tab = path_gdb_out + name_tgb_par_tab
arcpy.JoinField_management(path_tgb_s, 'tgb', path_tgb_par_tab, 'TGB')

# %% utgb.dat: impervious area, land use, and soil parameter file
# calculate HRUs' parameters based on selected GIS data and methods
df_data_utgbdat = tc.calc_hrus(
        path_tgb_s, path_soil, path_lu, 'tgb', f_lu_id,
        lu_id_imp, lu_id_water, def_amin_utgb_del,
        path_gdb_out, name_hru_c=name_hru_c, 
        ctrl_opt_impperc=ctrl_opt_impperc, f_impperc=f_impperc, 
        ctrl_opt_infdyn=ctrl_opt_infdyn, df_lu_mp=df_lu_mp,
        ctrl_opt_siltup=ctrl_opt_siltup,
        ctrl_opt_capr=ctrl_opt_capr, 
        ctrl_hru_aggr=ctrl_hru_aggr,
        print_out=print_out)
# export parameters to table
tc.df_to_table(df_data_utgbdat, path_gdb_out, name_utgb_par_tab)
# write utgb.dat file
path_utgbdat = path_files_out + name_utgbdat
tc.write_utgbdat(df_data_utgbdat, path_utgbdat, 
                 ctrl_opt_infdyn, ctrl_opt_impperc, ctrl_opt_capr, ctrl_opt_siltup,
                 udef_tgb_nodata_val=def_utgb_nodata_val, src_geodata=src_geodata,
                 catch_name=catch_name, comment=utgbdat_comment, 
                 print_out=print_out)

# %% profile.dat: external cross section implementation
# create cross section profile parameter file if option is active
if ctrl_rout_cs_file:
    # create cross section lines
    path_fa_hr = path_gdb_out + name_fa_hr
    tr.create_csl(path_fnw, path_ws_s, path_fa_hr, def_cs_dist, def_cs_wmax_eval,
                  path_gdb_out, name_csl=name_csl, name_fnw_o=name_fnw_o,
                  def_river_a_in=def_river_a_in, def_cs_search_rad=def_cs_search_rad,
                  print_out=print_out)
    
# --> manipulate cross sections
#     1. load in GIS software:
#        a. the model network polyline feature class (name_mnw) 
#        b. the polygon feature class representing the model raster 
#        c. the cross section polyline feature class
#        d. the digital elevation model
#     2. manipulate the cross sections to fulfill the following requirements:
#        a. only one cross section is allowed per model element
#        b. the cross section mid-point (intersection with flow network) has to
#           be within a routing cell (no headwater element!)
#        c. recommended: the cross section should capture a section of the valley
#           that is representative for the model element it is located in
#        d. recommended: intersections of cross sections should be avoided.
#        You may do the following:
#        a. turn the cross section
#        b. move the cross section (mid-point has to stay snapped to the flow
#           network polyline element with the same OBJECTID)
#        c. delete the cross section
    
    # create profile.dat based on user-defined cross sections
    # get inputs
    ser_pef_bm = df_data_tgbdat.loc[:, 'HM']
    ser_pef_hm = df_data_tgbdat.loc[:, 'BM']
    ser_tgb_down = df_data_tgb_p.loc[:, 'tgb_down']
    ser_tgb_type_routing = df_data_tgb_p.loc[:, 'tgb_type'] == 'routing'
    # create profile.dat based on user-defined cross sections
    path_fnw_o = path_gdb_out + name_fnw_o
    path_csl   = path_gdb_out + name_csl
    df_profdat_par, ser_tgb_csl = tr.df_profdat_from_cs(
            path_fnw, path_fnw_o, path_csl, path_dem_hr, path_fa_hr, path_tgb_s, 
            ser_tgb_down, ser_tgb_down_nd, ser_tgb_type_routing,
            ser_ch_form_q, ser_pef_bm, ser_pef_hm, 
            path_gdb_out, name_profile_par=name_profile_par,
            name_ch_fit_coords=name_ch_fit_coords, name_bcwsl=name_bcwsl, 
            def_cs_wmax_eval=def_cs_wmax_eval,
            def_cs_intp_buf_dist=def_cs_intp_buf_dist,
            def_ch_w=def_ch_w, def_ch_h=def_ch_h,
            def_ch_wres=def_ch_wres, def_cs_hmax_eval=def_cs_hmax_eval, 
            def_lam_hres=def_lam_hres,
            def_ch_vmin=def_ch_vmin, def_ch_vmax=def_ch_vmax,
            def_ch_wmax_eval=def_ch_wmax_eval, 
            def_chbank_slmin=def_chbank_slmin, def_ch_hmin=def_ch_hmin,
            def_ch_hmin_eval=def_ch_hmin_eval, def_profdat_decmax=def_profdat_decmax, 
            ctrl_show_plots=ctrl_show_plots, ctrl_save_plots=ctrl_save_plots, 
            ser_ft=ser_ft, ser_area_outfl=ser_area_outfl,
            def_ch_hmax_eval=def_ch_hmax_eval, path_plots_out=path_plots_out,
            print_out=print_out)
    # write profile.dat file
    path_profdat = path_files_out + name_profdat
    tc.write_profdat(df_profdat_par, ser_tgb_csl, path_profdat, 
                     def_cs_hmax_eval, def_lam_hres, 
                     def_profdat_nodata_val=def_profdat_nodata_val,
                     def_profdat_exit_val=def_profdat_exit_val, 
                     src_geodata=src_geodata, catch_name=catch_name,
                     comment=profdat_comment, 
                     print_out=print_out)    