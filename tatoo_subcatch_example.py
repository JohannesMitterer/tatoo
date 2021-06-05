# -*- coding: utf-8 -*-
"""
This module contains an calculation example to produce the spatial files
of a LARSIM subcatchment model (tgb.dat, utgb.dat, profile.dat).
It uses functions of the libraries 'TATOO core' and 'TATOO subcatchment'.

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
import tatoo_subcatch as tsc
import tatoo_common as tc

# %% control section
# define input layers
# geodatabase where inputs are stored
path_gdb_in       = r'a:\tatoo\subcatch_example\input.gdb' + '\\'
path_dem          = path_gdb_in + 'dem' # high resolution dem (pre-processed with burned river network)
path_fnw          = path_gdb_in + 'fnw' # flow network polyline feature class
path_lu           = path_gdb_in + 'landuse' # polygon feature class of land use data and sealing percentage
path_soil         = path_gdb_in + 'soil' # polygon feature class of soil data
# define output layers
# geodatabase where outputs are stored
path_gdb_out      = r'a:\tatoo\subcatch_example\output.gdb' + '\\'
name_ws_s         = 'ws_s'         # model domain polygon feature class
name_tgb_sj       = 'tgb_sj'       # model element polygon feature class
name_tgb_p        = 'tgb_p'        # model element point feature class
name_tgb_par_tab  = 'tgb_par_tab'  # model element table with output file parameters
name_hru_c        = 'hru_c'        # polygon feature class of HRUs
name_utgb_par_tab = 'utgb_par_tab' # model element ArcGIS table with output file parameters
# define output file names
# path where intermediate and final files are stored
path_files_out    = r'a:\tatoo\subcatch_example\model_files' + '\\' 
name_tgbdat       = 'tgb.dat'      # name of the output file (LARSIM requires a file named tgb.dat)
name_utgbdat      = 'utgb.dat'     # name of the output file (LARSIM requires a file named 'utgb.dat')

# define parameters for subcatchment optimization
# deepening value at flow network
h_burn             =  10      # [m]
# optimal model cell area
def_sc_area        = 1000000  # [m²]
# tolerance for flow length mismatch for catchment optimization
def_fl_min_tol     =  50      # [m]
# minimum subcatchment area tolerated
def_a_min_tol      =  50      # [m²]

# define parameters for runoff concentration processing
# define minimum slope (LARSIM internal convention: max. 4 decimals, but ~= 0)
def_sl_min         =   0.0001 # [mZ/mL]
# set default flow length for structural mismatch and negative transition deviations
# attention: 1 is interpreted by LARSIM as 'no routing'!
def_fl_strct_mism  =   2      # [m]
# set upper threshold for realistic flow length
def_fl_up_thr_fac  = 300      # [m]
# set upper threshold for realistic slope as quantile
def_sl_excl_quant  =   0.999  # [-]
# define HUT for cells with stream as percentage of stream's low and high point
def_zmin_rout_fac  =   0.5    # [-]
# define HOT as highest point of cell (as defined by Kirpich 1940)
def_zmax_fac       =   1      # [-]

# define parameters for channel estimation
# (de-)activate fitting of triple trapezoid cross section profiles
ctrl_rout_cs_fit   = True
# Linear distance between two automatically derived, consecutive cross sections
def_cs_dist_eval   =  50      # [m]
# maximum width of cross section evaluation
def_cs_wmax_eval   = 600      # [m]
# maximum height of cross section evaluation
def_cs_hmax_eval   =  10      # [m]
# estimated maximum flood plain width
def_flpl_wmax_eval =  50      # [m]
# maximum width of channel evaluation in [m]
def_ch_wmax_eval   =  40      # [m]
# minimum and maximum height of channel evaluation in [m]
def_ch_hmin_eval   =   0.1    # [m]
def_ch_hmax_eval   =   3.0    # [m]
# minimum channel depth threshold for channel identification, must be >= 0.2
def_ch_hmin        =   0.2    # [m]
# minimum reasonable flow velocity
def_ch_vmin        =   0.5    # [m/s]
# maximum reasonable flow velocity
def_ch_vmax        =   3.0    # [m/s]
# minimum channel bank slope threshold dH/dL for channel identification
def_chbank_slmin   =   0.1    # [-]
# horizontal resolution of interpolated points within valley
def_val_wres       =  10      # [m]
# horizontal resolution of interpolated points within flood plain
def_flpl_wres      =   0.2    # [m]
# artificial channel width and depth, added to continuiously descending cross sections
def_ch_w           =   0.5    # [m]
def_ch_h           =   0.5    # [m]
# horizontal resolution of interpolated points within channel
def_ch_wres        =   0.05   # [m]
# vertical spacing between evaluation lamellae
def_lam_hres       =   0.1    # [m]
# (de-)activate plot output
ctrl_show_plots    = False    # show plots?
ctrl_save_plots    = True     # save plots?
# define path, where figures shall be saved
path_plots_out     = path_files_out + r'fig' + '\\'

# define parameters for geomorphology
hq_ch             =   8.0     # [m³/s] HQ2 (or MHQ summer) as float
hq_ch_a           = 175.5     # [km²] catchment area as float
# ascending order of inflow cell IDs [-] as list of integers (index) and
# HQ2 (or MHQ) of inflow cells [m³/s] as list of floats (data)
# (e.g., pd.Series(np.array([34.5, 3.2]), index=[346, 687], name='q_in'))
ser_q_in_corr     = pd.Series(np.array([]), index=[], name='q_in')
# string defining channel estimation function
# may be 'Allen', 'Krauter' or 'combined'
ch_est_method     = 'combined'
# foreland edge default width
def_bx            =   0       # [m] (0 = not existing)
# foreland default factor (bbl = bm * def_bbx_fac, bbr = bm * def_bbx_fac)
def_bbx_fac       =   1       # [-]
# channel slope
def_bnm           =   1.5     # [mL/mZ] (1.5 = 67% slope)
# foreland slope
def_bnx           = 100       # [mL/mZ] (100 = 1% slope, ~ even)
# foreland boarder slope
def_bnvrx         =   4       # [mL/mZ] (4 = 25% slope)
# rouighness values
def_skm           =  30       # [m1/3s-1] (natural river channel, vegetated river bank)
def_skx           =  20       # [m1/3s-1] (uneven vegetated foreland)

# define parameters for hydrological response unit handling
# control operator to activate dynamic infiltration routine parametrization
ctrl_opt_infdyn   = True      # False: no parametrization, True: parametrization
# control operator to activate sealing parametrization in utgb.dat
ctrl_opt_impperc  = True      # False: no parametrization, True: parametrization
# control operator to activate capillary rise parametrization in utgb.dat
ctrl_opt_capr     = False     # False: no parametrization, True: parametrization
# control operator to activate silting-up parametrization in utgb.dat
ctrl_opt_siltup   = True      # False: no parametrization, True: parametrization
# control operator to define HRU aggregation
ctrl_hru_aggr     = True      # False: no aggregation, True: aggregate
# define area threshold below which HRUs are deleted
def_amin_utgb_del = 10**-8    # [m²]

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
lu_id_imp           =     3 # impervious land use class
lu_id_water         =    16 # water land use class

# define parameters for all model files
# catchment name (one line string, print in header of file)
catchment_name      = 'Alteneich'
# source of used geoinformation (one line string, print in header of file)
src_geodata         = 'Bavarian Environmental Agency (LfU)'

# define parameters for model element parameter file tgb.dat
# comment (one line string, print in header of file)
tgbdat_comment      = 'Oberstimm / Brautlach (13249007) HQ2 (Alteneich: 14.8 km²)'
# horizontal coordinate system EPDG number
hcs_epsg            = 25833 # [-]
# vertical coordinate system units
vcs_unit            = 'm ue. NN'
# no data value of the tgb.dat file (max. 3 characters)
def_tgb_nodata_val  =    -1 # [-]

# define parameters for model element parameter file utgb.dat
# comment (one line string, print in header of file)
utgbdat_comment     = 'Oberstimm / Brautlach (13249007) HQ2 (Alteneich: 14.8 km²)'
# no data value of the utgb.dat file (max. 3 characters)
def_utgb_nodata_val =    -1 # [-]

# swith command line output on (True) or off (False)
print_out           = True

# %% calculation section
# set intermediate feature class and raster names
name_pp      = 'pp_ws'          # pour point feature class
name_dem_c   = 'dem_c'          # dem clipped to catchment extent
name_fd_c    = 'fd_c'           # flow direction
name_fa_c    = 'fa_c'           # flow accumulation
name_fl_c    = 'fl_c'           # flow length
name_ws_s    = 'ws_s'           # catchment polygon feature class
name_pp_sc   = 'pp_sc'          # subcatchment pour points' feature class
name_sc_ws_s = 'sc_ws_s'        # pour point subbasin polygon feature class
name_fnw_fa  = 'fnw_fa'         # flow accumulation network polyline feature class
# define field names
f_pp_ws      = 'ModelWatershed' # watershed numbers (in name_pp)

# %% tgb.dat: model structure file including runoff concentration and routing parameters
# and cross section fitting (if activated)

# create pour point feature class
tc.create_pourpoint(path_fnw, path_gdb_out, name_pp=name_pp,
                    field_pp=f_pp_ws, print_out=print_out)

# --> manually create pour point features in ArcGIS
#     1. load the digital elevation model (name_dem) in GIS software
#     2. create pour point features defining outflow points of
#        a. the watershed defining the model domain
#        b. upstream watersheds, which shall be excluded from the model domain
#     3. define integer numbers in the variable 'ModelWatershed'
#        a. using positive integers for watersheds within the model domain (i.e., 2)
#        b. using negative integers for watersheds, which shall be excluded (i.e., -1)

# calculate subcatchments optimizing pour points according to specifications
path_pp_ws   = path_gdb_out + name_pp
pp_points_df = tsc.optimize_subc(path_dem, path_fnw, path_pp_ws, f_pp_ws,
                             def_sc_area, def_fl_min_tol, 
                             path_gdb_out, path_files_out, 
                             name_dem_c=name_dem_c, name_fd_c=name_fd_c, 
                             name_fa_c=name_fa_c, name_fl_c=name_fl_c, 
                             name_ws_s=name_ws_s, name_pp_sc=name_pp_sc, 
                             name_sc_ws_s=name_sc_ws_s, name_fnw_fa=name_fnw_fa,
                             h_burn=h_burn,
                             print_out=print_out)

# --> control optimized subcatchments in ArcGIS
#     1. load the following feature classes in GIS software
#        a. ws_s:    overall model domain watershed
#        b. pp_sc:   optimized pour points of subcatchments
#        c. sc_ws_s: watershed polygons of optimized subcatchments
#        d. fnw_fa:  flow path network calculated from flow accumulation
#     2. check size and position of subcatchment polygons
#     3. if neccessary, adapt position of created pour points using "edit"-tools
#        you may move, create or delete points.
#        Attention: Keep care of the ID numbers. They have to be unique in the end.

# calculate spatial parameters in a pd.DataFrame for export of the tgb.dat file
path_dem_c = path_gdb_out + name_dem_c
path_fd_c  = path_gdb_out + name_fd_c
path_fa_c  = path_gdb_out + name_fa_c
path_fl_c  = path_gdb_out + name_fl_c
path_pp_sc = path_gdb_out + name_pp_sc
df_data_tgbdat = tsc.calc_params(
        hq_ch, hq_ch_a,
        path_dem_c, path_fd_c, path_fa_c, path_fl_c, path_pp_sc, 
        path_files_out, path_gdb_out,
        name_tgb_p=name_tgb_p, name_tgb_sj=name_tgb_sj,
        def_sc_area=def_sc_area, def_a_min_tol=def_a_min_tol, 
        def_zmin_rout_fac=def_zmin_rout_fac, def_sl_excl_quant=def_sl_excl_quant, 
        ch_est_method=ch_est_method, ser_q_in_corr=ser_q_in_corr, 
        def_bx=def_bx, def_bbx_fac=def_bbx_fac, def_bnm=def_bnm,
        def_bnx=def_bnx, def_bnvrx=def_bnvrx, def_skm=def_skm,
        def_skx=def_skx, 
        ctrl_rout_cs_fit=ctrl_rout_cs_fit,
        def_cs_dist_eval=def_cs_dist_eval, 
        def_cs_wmax_eval=def_cs_wmax_eval, def_cs_hmax_eval=def_cs_hmax_eval,
        def_flpl_wmax_eval=def_flpl_wmax_eval,
        def_ch_wmax_eval=def_ch_wmax_eval, def_ch_hmin_eval=def_ch_hmin_eval,
        def_ch_hmax_eval=def_ch_hmax_eval,
        def_ch_hmin=def_ch_hmin, def_ch_vmin=def_ch_vmin,
        def_ch_vmax=def_ch_vmax, def_chbank_slmin=def_chbank_slmin,
        def_val_wres=def_val_wres, def_flpl_wres=def_flpl_wres,
        def_ch_w=def_ch_w, def_ch_h=def_ch_h, def_ch_wres=def_ch_wres,
        def_lam_hres=def_lam_hres,
        ctrl_show_plots=ctrl_show_plots, ctrl_save_plots=ctrl_save_plots,
        path_plots_out=path_plots_out, print_out=print_out)
# write tgb.dat file
path_tgbdat = path_files_out + name_tgbdat
tc.write_tgbdat(df_data_tgbdat, path_tgbdat, def_tgb_nodata_val=def_tgb_nodata_val, 
             hcs_epsg=hcs_epsg, vcs_unit=vcs_unit,
             src_geodata=src_geodata, catch_name=catchment_name, 
             comment=tgbdat_comment, print_out=print_out)
# join fields to polygon feature class of model elements
path_tgb_sj = path_gdb_out + name_tgb_sj
path_tgb_par_tab = path_gdb_out + name_tgb_par_tab
arcpy.JoinField_management(path_tgb_sj, 'tgb', path_tgb_par_tab, 'TGB')

# %% utgb.dat: impervious area, land use, and soil parameter file
# calculate HRUs' parameters based on selected GIS data and methods
df_data_utgbdat = tc.calc_hrus(
        path_tgb_sj, path_soil, path_lu, 'tgb', f_lu_id,
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
                 catch_name=catchment_name, comment=utgbdat_comment, 
                 print_out=print_out)

