# -*- coding: utf-8 -*-
"""
This library contains all functions needed to produce the spatial files
of a LARSIM raster model (tgb.dat, utgb.dat, profile.dat).
It uses functions from the TATOO core library.

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

# load modules
import os
import sys
import copy
import arcpy
import numpy.matlib
import numpy.lib.recfunctions
import numpy as np
import pandas as pd
import tatoo_common as tc

# check out ArcGIS spatial analyst license
class LicenseError(Exception):
    pass
try:
    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
        print ("Checked out \"Spatial\" Extension")
    else:
        raise LicenseError
except LicenseError:
    print("Spatial Analyst license is unavailable")
except:
    print(arcpy.GetMessages(2))

# allow overwriting the outputs
arcpy.env.overwriteOutput = True

# %% function to preprocess a high-resolution digital elevation model
def preprocess_dem(path_dem_hr, path_fnw, cellsz, h_burn,
                   path_gdb_out, name_dem_mr_f='dem_mr_f',
                   print_out=False):
    """
    Aggregates a high-resolution digital elevation raster, covnert river network 
    to model resolution raster, burns flow network raster into digital elevation
    raster and fills sinks of the resulting raster.

    JM 2021

    Arguments:
    -----------
    path_dem_hr: str
        path of the high-resolution digital elevation raster
        (e.g., 'c:\model_creation.gdb\dem_hr')
    path_fnw: str
        path of the flow network feature class or shape file (e.g., 'c:\fnw.shp')
    cellsz: integer
        edge length of the resulting model cells in [m] (e.g., 100)
    h_burn: integer
        depth of river network burning in digital elevation model
    path_gdb_out: str
        path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
    name_dem_mr_f: str
        name of the filled model-resolution digital elevation raster(e.g., 'dem_mr_f')
    print_out: boolean (optional, default: False)
        true if workprogress shall be print to command line (default: false)

    Returns:
    -----------
    Saves the following files:
        - filled model-resolution digital elevation raster
        - model-resolution raster representation of the flow network
    
    """
    # arcpy field names (ONLY CHANGE IF ARCPY FUNCTION CHANGES FORCE YOU!)
    f_oid = 'OBJECTID'
    # paths for intermediates
    path_dem_mr_sr   = path_gdb_out + 'dem_mr_sr'
    path_dem_mr      = path_gdb_out + 'dem_mr'
    path_fnw_mr      = path_gdb_out + 'fnw_mr'
    path_dem_mr_cfnw = path_gdb_out + 'dem_mr_cfnw'
    # paths for outputs
    path_dem_mr_f = path_gdb_out + name_dem_mr_f
    # Aggregate high resolution digital elevation model to model resolution
    if print_out: print('...aggregate high resolution digital elevation model...')
    # create snap raster at origin of coordinate system
    dem_mr_sr = arcpy.sa.CreateConstantRaster(
            1, 'INTEGER', cellsz, arcpy.Extent(
                             0.5 * cellsz,          0.5 * cellsz,
                    cellsz + 0.5 * cellsz, cellsz + 0.5 * cellsz))
    dem_mr_sr.save(path_dem_mr_sr)
    # save default and set environments
    default_env_snr      = arcpy.env.snapRaster
    default_env_ext      = arcpy.env.extent
    arcpy.env.snapRaster = path_dem_mr_sr
    arcpy.env.extent     = path_dem_hr
    # aggregate high resolution DEM to model resolution
    if arcpy.Exists(path_dem_mr):
        arcpy.management.Delete(path_dem_mr)
    dem_mr = arcpy.sa.Aggregate(path_dem_hr, cellsz, 'MEAN', 'EXPAND', 'DATA')
    dem_mr.save(path_dem_mr)
    
    # cut rivers
    if print_out: print('...cut rivers...')
    # convert polylines to raster in model grid size
    arcpy.conversion.PolylineToRaster(path_fnw, f_oid, path_fnw_mr,
                                      'MAXIMUM_LENGTH', 'NONE', path_dem_mr)
    # decrease model resolution elevation raster values at flow network raster cells
    dem_mr_cfnw = arcpy.sa.Con(arcpy.sa.IsNull(path_fnw_mr), path_dem_mr,
                               dem_mr - h_burn)
    dem_mr_cfnw.save(path_dem_mr_cfnw)
    # reset environment parameters
    arcpy.env.snapRaster = default_env_snr
    arcpy.env.extent     = default_env_ext
    
    # fill cut model resolution digital elevation raster sinks
    if print_out: print('...fill cut model resolution digital elevation raster sinks...')
    # fill sinks
    dem_mr_cfnw_f = arcpy.sa.Fill(path_dem_mr_cfnw, '')
    dem_mr_cfnw_f.save(path_dem_mr_f)

# %% function to calculate the model watershed
def calc_watershed(path_dem_mr_cfnw_f, path_pp,
                   path_gdb_out, name_fd_mr='fd_mr', name_fd_mr_corr='fd_mr_corr',
                   name_fa_mr='fa_mr', name_ws_s='ws_s', 
                   initial=True, name_fd_p_corr='fd_p_corr', path_fd_p_corr='', 
                   print_out=False):
    """
    Creates the model watershed from a filled digital elevation raster using
    pour points. The correction point feature class is necessary for the initial
    calculation of the model watershed calculation.

    JM 2021

    Arguments:
    -----------
    path_dem_mr_cfnw_f: str
        path of the filled model-resolution digital elevation raster
        (e.g., 'c:\model_creation.gdb\dem_mr_cfnw_f')
    path_pp: str
        path of the pour point feature class
        (e.g., 'c:\model_creation.gdb\pp')
    path_gdb_out: str
        path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
    name_fd_mr: str
        name of the output model resolution flow direction raster (e.g., 'fd_mr')
    name_fd_mr: str
        name of the corrected output model resolution flow direction raster
        (e.g., 'fd_mr_corr')
    name_fa_mr: str
        name of the output model resolution flow accumulation raster (e.g., 'fa_mr')
    name_ws_s: str
        name of the output watershed polygon feature class (e.g., 'ws_s')
    initial: boolean (optional)
        true if it is the initial run to calculate the model watershed
    name_fd_p_corr: str (optional)
        name of the output flow direction correction point feature class
        (e.g., 'fd_p_corr')
    path_fd_p_corr: str (optional)
        path of the output flow direction correction point feature class
        needed for case initial=False (e.g., 'fd_p_corr')
    print_out: boolean (optional, default: False)
        true if workprogress shall be print to command line

    Returns:
    -----------
    Saves the following outputs:
        - model resolution flow direction raster
        - watershed polygon feature class
        - flow direction correction point feature class (optional)
    
    """
    # check inputs
    if not initial and not path_fd_p_corr:
        sys.exit('With initial=False path_fd_p_corr must not be an empty string!')
    # define internal field names
    f_pp    = 'pp'
    f_pp_ws = 'ModelWatershed'
    # arcpy field names (ONLY CHANGE IF ARCPY FUNCTION CHANGES FORCE YOU!)
    f_oid = 'OBJECTID'
    f_val = 'Value'
    f_VAL = 'VALUE'
    # feature class names from input
    name_pp = os.path.split(path_pp)[1]
    # define paths of intermediates in working geodatabase
    path_ppc     = path_gdb_out + 'ppc'
    path_spp     = path_gdb_out + 'spp'
    path_ws_r    = path_gdb_out + 'ws_r'
    path_ws_sr   = path_gdb_out + 'ws_sr'
    path_ws_s_sj = path_gdb_out + 'ws_s_sj'
    if not initial: path_fd_r_corr = path_gdb_out + 'fd_r_corr'
    # paths for outputs
    path_fd_mr      = path_gdb_out + name_fd_mr
    path_fd_mr_corr = path_gdb_out + name_fd_mr_corr
    path_fa_mr      = path_gdb_out + name_fa_mr
    path_ws_s       = path_gdb_out + name_ws_s
    if initial: path_fd_p_corr = path_gdb_out + name_fd_p_corr
    
    # calculate flow direction
    if print_out: print('...calculate flow direction raster...')
    if arcpy.Exists(path_fd_mr): arcpy.management.Delete(path_fd_mr)
    fd_mr = arcpy.sa.FlowDirection(path_dem_mr_cfnw_f, 'NORMAL', '', 'D8')
    fd_mr.save(path_fd_mr)
    
    # if run is initial, create correction flow direction point feature class
    # and copy flow direction raster.
    field_fd_corr = 'D8'
    if initial:
        # create flow direction correction feature class and add flow direction
        # binary field
        sr = arcpy.Describe(path_pp).spatialReference
        arcpy.CreateFeatureclass_management(path_gdb_out, name_fd_p_corr,
                                            'POINT', '', 'DISABLED', 'DISABLED',
                                            sr, '', '0', '0', '0', '')
        arcpy.AddField_management(path_fd_p_corr, field_fd_corr, 'SHORT', '', '',
                                  '', '', 'NULLABLE', 'NON_REQUIRED', '')
        if arcpy.Exists(path_fd_mr_corr): arcpy.management.Delete(path_fd_mr_corr)
        arcpy.CopyRaster_management(path_fd_mr, path_fd_mr_corr, '', '', '255',
                                    'NONE', 'NONE', '8_BIT_UNSIGNED', 'NONE',
                                    'NONE', 'GRID', 'NONE', 'CURRENT_SLICE',
                                    'NO_TRANSPOSE')
    # else, correct flow direction raster using correction point features
    else:
        # get number of existing flow direction correction point features
        fd_p_corr_descr = arcpy.Describe(path_fd_p_corr)
        fd_p_nb = fd_p_corr_descr.extent.XMin
        # if there are existing point features (nb!=0), do correction
        if not np.isnan(fd_p_nb):
            # set environments
            default_env_snr      = arcpy.env.snapRaster
            default_env_csz      = arcpy.env.cellSize
            arcpy.env.snapRaster = path_fd_mr
            arcpy.env.cellSize   = path_fd_mr
            # convert flow direction correction points to raster
            arcpy.PointToRaster_conversion(path_fd_p_corr, field_fd_corr,
                                           path_fd_r_corr, 'MOST_FREQUENT',
                                           'NONE', path_fd_mr)
            # change environments
            default_env_ext  = arcpy.env.extent
            default_env_mask = arcpy.env.mask
            arcpy.env.extent = 'MAXOF'
            arcpy.env.mask   = path_fd_mr
            # replace flow direction values, where correction points are defined
            fd_mr_corr = arcpy.ia.Con(arcpy.ia.IsNull(path_fd_r_corr), path_fd_mr,
                                      path_fd_r_corr)
            fd_mr_corr.save(path_fd_mr_corr)
            # reset environments
            arcpy.env.snapRaster = default_env_snr
            arcpy.env.cellSize   = default_env_csz
            arcpy.env.extent     = default_env_ext
            arcpy.env.mask       = default_env_mask
        # else, copy uncorrected flow direction raster
        else:
            print(('INFO: Flow direction correction point feature'
                   'class is empty. Original flow direction is used instead.'))
            if arcpy.Exists(path_fd_mr_corr): arcpy.management.Delete(path_fd_mr_corr)
            arcpy.CopyRaster_management(path_fd_mr, path_fd_mr_corr, '', '', '255',
                                        'NONE', 'NONE', '8_BIT_UNSIGNED', 'NONE',
                                        'NONE', 'GRID', 'NONE', 'CURRENT_SLICE',
                                        'NO_TRANSPOSE')
    
    if print_out: print('...calculate flow accumulation...')
    # calculate flow accumulation raster
    if arcpy.Exists(path_fa_mr): arcpy.management.Delete(path_fa_mr)
    fa_mr = arcpy.sa.FlowAccumulation(path_fd_mr_corr, '', 'DOUBLE', 'D8')
    fa_mr.save(path_fa_mr)
    # copy pour point feature class
    if arcpy.Exists(path_ppc): arcpy.management.Delete(path_ppc)
    arcpy.management.CopyFeatures(path_pp, path_ppc, '', '', '', '')
    # add adn calculate field using the object ID
    arcpy.AddField_management(path_ppc, f_pp, 'LONG', '', '', '', '', 'NULLABLE',
                              'NON_REQUIRED', '')
    arcpy.CalculateField_management(path_ppc, f_pp, '!{0}!'.format(f_oid), 'PYTHON3', '')
    # snap pour points to flow accumulation raster
    if arcpy.Exists(path_spp): arcpy.management.Delete(path_spp)
    spp = arcpy.sa.SnapPourPoint(path_ppc, fa_mr, '40', f_pp)
    spp.save(path_spp)
    
    if print_out: print('...calculate watershed...')
    # calculate watershed raster
    if arcpy.Exists(path_ws_r): arcpy.management.Delete(path_ws_r)
    ws_r = arcpy.sa.Watershed(path_fd_mr_corr, spp, f_val)
    ws_r.save(path_ws_r)
    # set environments
    arcpy.env.outputZFlag = 'Same As Input'
    arcpy.env.outputMFlag = 'Same As Input'
    # convert watershed raster to polygon features
    if arcpy.Exists(path_ws_sr): arcpy.management.Delete(path_ws_sr)
    arcpy.RasterToPolygon_conversion(path_ws_r, path_ws_sr, 'NO_SIMPLIFY', f_VAL,
                                     'SINGLE_OUTER_PART', '')
    
    if print_out: print('...select model watersheds...')
    pp_fieldnames = [field.name for field in arcpy.ListFields(path_pp)]
    # if field exists, that identifies polygon as model watershed, delete watersheds
    # with the fields' value >= 1
    if f_pp_ws in pp_fieldnames:
        # join created watershed polygons to pour points
        arcpy.SpatialJoin_analysis(
                path_ws_sr, path_pp, path_ws_s_sj, 'JOIN_ONE_TO_ONE', 'KEEP_ALL', 
                "{0} '{0}' true true false 2 Short 0 0,First,#,{1},{0},-1,-1".format(
                f_pp_ws, name_pp), 'CONTAINS', '', '')
        # select and copy model watersheds marked with a positive integer
        sel_sql = f_pp_ws + ' >= 1'
        path_ws_s_sj_sel = arcpy.management.SelectLayerByAttribute(
                path_ws_s_sj, 'NEW_SELECTION', sel_sql)
        if arcpy.Exists(path_ws_s): arcpy.management.Delete(path_ws_s)
        arcpy.management.CopyFeatures(path_ws_s_sj_sel, path_ws_s, '', '', '', '')
    else:
        if arcpy.Exists(path_ws_s): arcpy.management.Delete(path_ws_s)
        arcpy.management.CopyFeatures(path_ws_sr, path_ws_s, '', '', '', '')

# %% function to calculate the model cell network
def calc_model_network(path_ws_s, path_fd_mr, path_fa_mr, 
                       path_gdb_out, path_files_out, name_fl_mr='fl_mr',
                       name_tgb_p='tgb_p', name_mnw='mwn', 
                       print_out=False):
    """
    Creates a point feature class representing the center of model cells
    as well as a polyline feature class representing the model network between
    the model cells (upstream-downstream-relation).

    JM 2021

    Arguments:
    -----------
    path_ws_s: str
        path of the output watershed polygon feature class
        (e.g., 'ws_s')
    path_fd_mr: str
        path of the output model resolution flow direction raster
        (e.g., 'fd_mr')
    path_fa_mr: str
        path of the output model resolution flow accumulation raster
        (e.g., 'fa_mr')
    path_gdb_out: str
        path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
    path_files_out: str
        storage path for intermediate data (e.g., 'c:\tmp_model_data\')
    name_fl_mr: str
        name of the extracted output model resolution flow length raster
        (e.g., 'fl_mr_e')
    name_tgb_p: str = (optional)
        name of the output model cell point feature class (e.g., 'tgb_p')
    name_tgbd_p: str = (optional)
        name of the output downstream model cell point feature class
        (e.g., 'tgbd_p')
    name_mnw: str = (optional)
        name of the output model network polyline feature class (e.g., 'mwn')
    print_out: boolean (optional)
        true if workprogress shall be print to command line

    Returns:
    -----------
    df_data_tgb_p: pd.DataFrame
        - tgb: model element ID number (int)
        - tgb_down: downstream model element ID number (int)
        - tgb_type: model element type (str)
        - tgb_dtgb: real representative model element ID for dummy elements (int)
        - tgb_a: inflowing catchment area of each model element [km²]
        - x, y: x- and y-coordinates of element center [m]
    df_tgb_up: pd.DataFrame
        tgb_up1, tgb_up2: upstream model element ID numbers (int)
    Saves the following outputs:
        - extracted model resolution flow length raster
        - model cell point feature class
        - downstream model cell point feature class
        - model network polyline feature class
    
    """
    # define internal variables
    def_val_dtgb = -1
    # define internal field names
    f_tgb      = 'tgb'
    f_tgb_down = 'tgb_down'
    f_tgb_type = 'tgb_type'
    f_tgb_dtgb = 'tgb_dtgb'
    f_tgb_a    = 'tgb_a'
    f_x        = 'x'
    f_y        = 'y'
    f_nrflv    = 'nrflv'
    f_tgb_up1  = 'up1'
    f_tgb_up2  = 'up2'
    # define key-words to identify element types
    str_headw   = 'headwater'
    str_routing = 'routing'
    str_dummy   = 'dummy'
    # arcpy field names (ONLY CHANGE IF ARCPY FUNCTION CHANGES FORCE YOU!)
    f_p_x = 'POINT_X'
    f_p_y = 'POINT_Y'
    # define paths of intermediates in working geodatabase
    path_fd_mr_e = path_gdb_out + 'fd_mr_e'
    path_fa_mr_e = path_gdb_out + 'fa_mr_e'
    name_tgb_down_p = 'tgb_down_p'
    # paths for outputs
    path_fl_mr_e = path_gdb_out + name_fl_mr
    path_mnw     = path_gdb_out + name_mnw
    
    # create real representative index list for dummy subcatchments
    def real_repr_idx(df_tgb, str_dummy, print_out=False):
        
        if print_out: print(('...create representative index list for '
                             'dummy subcatchments tgb_dtgb...'))
        # Preallocate arrays
        ser_tgb_dtgb = pd.Series(np.ones(df_tgb.shape[0]) * def_val_dtgb,
                         index=df_tgb.index, name=f_tgb_dtgb).astype(np.int)
        # Iterate over all final index values
        for tgb in df_tgb.index:
            # if cell is a dummy, find the connected real cell
            if df_tgb.at[tgb, f_tgb_type] == str_dummy:
                # follow dummy cascade downwards until real cell and set index
                mm = copy.deepcopy(tgb)
                while df_tgb.at[mm, f_tgb_type] == str_dummy:
                    mm  = df_tgb.at[mm, f_tgb_down]
                ser_tgb_dtgb.at[tgb] = mm
        return ser_tgb_dtgb
    
    # calculations
    # (de-)activate additional debugging command line output
    debug = False # (False/True)
    # set workspace
    arcpy.env.workspace = path_gdb_out
    # clip flow direction raster to watershed polygon
    if print_out: print('...clip flow direction raster...')
    if arcpy.Exists(path_fd_mr_e): arcpy.management.Delete(path_fd_mr_e)
    fd_mr_e      = arcpy.sa.ExtractByMask(path_fd_mr, path_ws_s)
    fd_mr_e.save(path_fd_mr_e)
    # clip flow accumulation raster to watershed polygon
    if print_out: print('...clip flow accumulation raster...')
    if arcpy.Exists(path_fa_mr_e): arcpy.management.Delete(path_fa_mr_e)
    fa_mr_e      = arcpy.sa.ExtractByMask(path_fa_mr, path_ws_s)
    fa_mr_e.save(path_fa_mr_e)
    # calculate downstream flow length
    if print_out: print('...calculate flow length...')
    if arcpy.Exists(path_fl_mr_e): arcpy.management.Delete(path_fl_mr_e)
    fl_mr_e      = arcpy.sa.FlowLength(fd_mr_e, 'DOWNSTREAM', '')
    fl_mr_e.save(path_fl_mr_e)
    
    if print_out: print('...import flow rasters...')
    # define paths of intermediates in working folder
    path_fd_c_tif    = path_files_out + 'fd_c.tif'
    path_fa_c_tif    = path_files_out + 'fa_c.tif'
    path_fl_c_tif    = path_files_out + 'fl_c.tif'
    # import flow direction, accumulation and length as numpy rasters
    fd, ncols, nrows, cellsz, xll, yll, ctrl_tif_export = tc.fdal_raster_to_numpy(
            path_fd_mr_e, 'fd', path_fd_c_tif, True)
    fa, _, _, _, _, _, _ = tc.fdal_raster_to_numpy(
            path_fa_mr_e, 'fa', path_fa_c_tif, False)
    fl, _, _, _, _, _, _ = tc.fdal_raster_to_numpy(
            path_fl_mr_e, 'fl', path_fl_c_tif, True)
    # add a NaN boundary to all gis input data sets
    empty_row = np.zeros((1,     ncols)) * np.nan
    empty_col = np.zeros((nrows + 2, 1)) * np.nan
    fa = np.concatenate((empty_row, fa, empty_row), axis=0)
    fa = np.concatenate((empty_col, fa, empty_col), axis=1)
    fd = np.concatenate((empty_row, fd, empty_row), axis=0)
    fd = np.concatenate((empty_col, fd, empty_col), axis=1)
    fl = np.concatenate((empty_row, fl, empty_row), axis=0)
    fl = np.concatenate((empty_col, fl, empty_col), axis=1)
    # adjust gis parameters for new sizes
    ncols = ncols + 2
    nrows = nrows + 2
    xll = xll - cellsz
    yll = yll - cellsz
    # set default data type for calculations for efficient RAM usage
    if ncols * nrows <= 32767: np_type = np.int32
    else: np_type = np.int64
    # get indices and number of not-nan-data
    gis_notnans       = np.nonzero(~np.isnan(fd))
    gis_notnans_x     = gis_notnans[0]
    gis_notnans_y     = gis_notnans[1]
    gis_notnans_count = gis_notnans_x.shape[0]
    # create lookup table connecting flow direction int-values to array indices
    fd_lu = np.array([[  1, 0, 1], [  2, 1, 1], [  4, 1, 0],
                      [  8, 1,-1], [ 16, 0,-1], [ 32,-1,-1],
                      [ 64,-1, 0], [128,-1, 1]])
    # pre-allocate flow direction arrays
    fd_xd = np.empty((gis_notnans_count, 1), dtype=np_type)
    fd_yd = np.empty((gis_notnans_count, 1), dtype=np_type)
    # iterate flow direction int-values
    for ii in range(fd_lu.shape[0]):
        # get indices of not-nan flow direction values with fitting int-value
        fd_notnans_ii = fd[~np.isnan(fd)] == fd_lu[ii, 0]
        # set array x and y index at found indices
        fd_xd[fd_notnans_ii] = fd_lu[ii, 1]
        fd_yd[fd_notnans_ii] = fd_lu[ii, 2]
    # create vector of combined not-nan array and converted flow direction indices
    Jtm_down_xd = gis_notnans_x + np.int64(fd_xd[:, 0])
    Jtm_down_yd = gis_notnans_y + np.int64(fd_yd[:, 0])
    
    if print_out: print('...initialize arrays for iteration...')
    # create temporal index array with continuous number Jtm
    Jtm      = np.ones((nrows, ncols), dtype=np_type) * -1
    Jtm[gis_notnans_x, gis_notnans_y] = range(1, gis_notnans_count+1)
    # calculate temporal downstream cell array Jtm_down using flow direction indices.
    Jtm_down = np.ones((nrows, ncols),dtype=np_type) * -1
    Jtm_down[gis_notnans] = Jtm[Jtm_down_xd, Jtm_down_yd]
    # find the catchment outlet where no downstream index is set
    OFr      = np.nonzero(np.logical_and(Jtm != -1, Jtm_down == -1))
    # mark the outlet cell in Jtm_down with a zero
    Jtm_down[OFr] = 0
    # preallocate list for temporal upstream index calculation Jt_up
    Jt_up    = np.ones((gis_notnans_count, 7), dtype=np_type) * -1
    # iterate temporal upstream list
    for jt_ii, jt in enumerate(range(1, Jt_up.shape[0] + 1)):
        # find all rows in Jtm_down which do have jt as downstream cell
        verw = np.nonzero(Jtm_down == jt)
        # print subset in temporal upstream list Jt_up
        Jt_up[jt_ii, 0:verw[0].shape[0]] = Jtm[verw]
    # convert list to int
    Jt_up    = np.int32(Jt_up)
    # calculate sum of necessary dummy cells (which have >2 upstream cells)
    D_count  = np.nansum(Jt_up[:, 2:7] != -1)
    # calculate number of temporal index numbers jt
    Jt_count = Jt_up.shape[0]
    # calculate number of final indices j as sum of dummy and real cells
    J_count  = Jt_count + D_count
    # preallocate temporal downstream list Jt_down
    Jt_down  = np.ones((Jt_count, 1), dtype=np_type) * -1
    # iterate over temporal index jt and fill list
    for jt_ii, jt in enumerate(range(1, Jt_count+1)):
        # look for downstream cell from matrix
        Jt_down[jt_ii] = Jtm_down[Jtm == jt]
    # preallocate lists for final indices J, J_type and J_jt, final upstream 
    # and downstream lists J_up and J_down, and protocol list Done
    J_type  = J_count * [None]
    J       = np.array(range(1, J_count+1))
    J_up    = np.ones((J_count, 2),     dtype=np_type) * -1
    J_down  = np.ones((J_count,  ),     dtype=np_type) * -1
    J_jt    = np.ones((J_count, 1),     dtype=np_type) * -1
    Done    = np.ones((np.nanmax(Jtm)), dtype=np_type) * -1
    # calculate protocol list D_contr
    D_contr = np.nansum(Jt_up[:, 2:] != -1, 1)

    # calculate final flow network index lists J, J_down, J_up, J_type, X and Y
    # iterating from largest flow length downstream to outlet (tree-climbing algorithm)
    if print_out: print('''...calculate final flow network index lists...''')
    # find cell with largest flow length and its temporal index
    jt  = Jtm[fl == np.nanmax(fl)][0]
    jti = jt - 1
    # preset upstream subset (ss)
    ss  = Jt_up[jti, :]
    ss  = ss[ss != -1]
    ssi = ss - 1
    # calculate not done subset of upstream cell subset
    ssnotdone = ss[Done[ssi] == -1]
    # pre-set final index variable (0)
    jj = 0
    # debug protocol
    if debug and print_out:
        im_pos = np.nonzero(Jtm == jt)
        x      = im_pos[0]
        y      = im_pos[1]
        print('   Initial cell at pos: {0:d}/{1:d} ({2:d})'.format(x, y, jt))
    # while either outlet is not reached or not all upstream members are processed
    while jt != Jtm[OFr] or ssnotdone.shape[0] != 0:
        # case 1: HEADWATER CELL as ssnotnan is empty
        # -> create new index for headwater and move downwards
        if ss.shape[0] == 0:
            # increment final index, fill type and link lists
            jj          += 1
            jji          = jj - 1
            J_type[jji]  = str_headw
            J_jt[jji, 0] = jt
            # debug protocol
            if debug and print_out:
                print('j: {0:d}, pos: {1:d}/{2:d} = {3:d} -> {4:d}, {5:s} cell'.format(
                        jj, x, y, jt, Jt_down[jti, 0], str_headw))
            # set upstream cell to 0, mark cell as done and go downwards
            J_up[jji, 0] = 0
            Done[jti]    = 1
            jt           = Jt_down[jti, 0]
            jti          = jt - 1
            # debug protocol
            if debug and print_out:
                im_pos = np.nonzero(Jtm == jt)
                x      = im_pos[0]
                y      = im_pos[1]
                print('   -> down to {0:d}/{1:d} = {2:d}'.format(x, y, jt))
        else:
            # case 2: ROUTING CELL as all upstream cells are done
            # -> create new index for routing cell and move downwards
            if all(Done[ssi] == 1):
                # increment final index, fill type and link lists
                jj += 1
                jji = jj - 1
                J_type[jji]  = str_routing
                J_jt[jji, 0] = jt
                # define upstream cell subset and give position indices
                ssj = np.flatnonzero(np.any(J_jt == ss, 1))
                # if one or two upstream cells exist:
                # connect two real cells in Jt_up and Jt_down
                if ssj.shape[0] <= 2:
                    ssjl             = ssj.shape[0]
                    ssjtu            = Jt_up[jti, :ssjl]
                    ssjtu            = ssjtu[ssjtu != -1]
                    J_up[jji, :ssjl] = J[np.flatnonzero(np.any(J_jt == ssjtu, 1))]
                    J_down[ssj]      = jj
                # else if > 2 upstream cells exist:
                # connect 1 real and 1 dammy cell in Jt_up and Jt_down
                else:
                    real                      = J[np.amax(ssj)]
                    dummy                     = np.amax(J_down[ssj])
                    J_up[jji, :]              = [dummy, real]
                    J_down[[dummy-1, real-1]] = jj
                # debug protocol
                if debug and print_out:
                    pr_up = Jt_up[jti, :]
                    pr_up = pr_up[pr_up != -1]
                    print('''j: {0:d}, Pos: {1:d}/{2:d} = {3:d} -> {4:d}, 
                          Jt_up: {5:s}, {6:s} cell'''.format(
                          jj, x, y, jt, Jt_down[jt-1],
                          str(pr_up[~np.isnan(pr_up)])[1:-1], str_routing))
                # mark cell as done and go downwards
                Done[jti] = 1
                jt        = Jt_down[jti, 0]
                jti       = jt - 1
                # debug protocol
                if debug and print_out:
                    im_pos = np.nonzero(Jtm == jt)
                    x      = im_pos[0]
                    y      = im_pos[1]
                    print('   -> down to {0:d}/{1:d} = {2:d}'.format(x, y, jt))
            else:
                # case 3: DUMMY CELL as not all required dummy cells are
                # done but >= 2 upstream cells are done
                # -> create new index for dummy cell and move upwards to
                # the cell with the largest flow accumulation
                if np.sum(Done[ssi] != -1) >= 2:
                    # increment final index, fill type and link lists
                    jj         += 1
                    jji         = jj - 1
                    J_type[jji] = str_dummy
                    J_jt[jji,0] = 0
                    # define upstream cell subset and give position indices
                    ssj = np.flatnonzero(J_down[0:jji] == -1)
                    # preallocate testing matrix (all are false)
                    ssjt = np.zeros((ssj.shape[0], ), dtype=bool)
                    # iterate upstream cell subset
                    for ii, ssji in enumerate(ssj):
                        jtupi = Jt_up[jti, :]
                        jtupi = jtupi[jtupi != -1]
                        # ssj exists in Jt_up -> test is TRUE
                        if np.any(np.isin(jtupi, J_jt[ssji, 0])):
                            ssjt[ii] = True
                        # ssj does not exist in Jt_up but is dummy
                        # -> test is TRUE
                        elif J_type[ssji] == str_dummy:
                            ssjt[ii] = True
                    # reduce subset with testing matrix
                    ssj = ssj[ssjt]
                    # 'wrong neighbours'
                    # (loose, not finished dummy strings) are removed
                    if ssj.shape[0] > 2:
                        ssj = ssj[-2:]
                    # connect upstream cells in Jt_up and Jt_down
                    J_up[jji, :] = J[ssj]
                    J_down[ssj]  = jj
                    # debug protocol
                    if debug and print_out:
                        pr_up = Jt_up[jti, :]
                        pr_up = pr_up[pr_up != -1]
                        print('''j: {0:d}, Pos: {1:d}/{2:d} = {3:d} -> {4:d}, 
                              Jt_up: {5:s}, {6:s} cell'''.format(
                              jj, x, y, jt, Jt_down[jti,0],
                              str(pr_up[~np.isnan(pr_up)])[1:-1], str_dummy))
                    # decrement dummy protocol variable
                    D_contr[jti] = D_contr[jti] - 1
                # case 4 (else): UPWARDS MOVEMENT as not all required dummy
                # cells are done and < 2 upstream cells are done
                # -> do not create new index
    
                # calculate not done subset of upstream cells and its largest
                # flow accumulation cell preallocate subset for flow 
                # accumulation calculation
                ssflowacc = np.zeros((ssnotdone.shape[0]), dtype=np_type)
                # iterate not done subset of upstream cells and find flow
                # accumulation
                for ii, iiv in enumerate(ssflowacc):
                    ssflowacc[ii] = fa[Jtm == ssnotdone[ii]]
                # calculate temporal index of max. flow accumulation 
                ssmaxind = ssnotdone[ssflowacc == np.amax(ssflowacc)]
                # go upstream to max flow acc or first cell if more than one
                # solutions exist
                jt  = ssmaxind[0]
                jti = jt - 1
                # debug protocol
                if debug and print_out:
                    im_pos = np.nonzero(Jtm == jt)
                    x      = im_pos[0]
                    y      = im_pos[1]
                    print('   -> up to {0:d}/{1:d} = {2:d}'.format(x, y, jt))
        # find upstream cells and create subset (ss)
        ss  = Jt_up[jti, :]
        ss  = ss[ss != -1]
        ssi = ss - 1
        # calculate not done subset of upstream cell subset
        ssnotdone = ss[Done[ssi] == -1]
        
    # Calculate values for catchment outlet
    if print_out: print('...calculate outlet...')
    # fill lists
    jj          += 1
    jji          = jj - 1
    J_jt[jji, 0] = jt
    J_type[jji]  = str_routing
    # debug protocol
    if debug and print_out:
        pr_up = Jt_up[jti, :]
        pr_up = pr_up[pr_up != -1]
        print('''j: {0:d}, Pos: {1:d}/{2:d} = {3:d} -> {4:d}, 
              Jt_up: {5:s}, {6:s} cell'''.format(
              jj, x, y, jt, Jt_down[jt-1],
              str(pr_up[~np.isnan(pr_up)])[1:-1], str_routing))
    # define upstream cell subset and give position indices
    ssj = np.flatnonzero(np.any(J_jt == ss, 1))
    # one or two upstream cells: connect two real cells in Jt_up and Jt_down
    if ssj.shape[0] <= 2:
        ssjl             = ssj.shape[0]
        ssjtu            = Jt_up[jti, :ssjl]
        ssjtu            = ssjtu[ssjtu != -1]
        J_up[jji, :ssjl] = J[np.flatnonzero(np.any(J_jt == ssjtu, 1))]
        J_down[ssj]      = jj
    # > 2 upstream cells: connect 1 real and 1 dammy cell in Jt_up and Jt_down
    else:
        real                      = J[np.amax(ssj)]
        dummy                     = np.amax(J_down[ssj])
        J_up[jji, :]              = [dummy, real]
        J_down[[dummy-1, real-1]] = jj
    # Define downstream cell as 0
    J_down[jji] = Jt_down[jti]
    
    # create final index array Jm and final dummy index list J_dj
    if print_out: print('...create final index array and final dummy index list...')
    # preallocate arrays
    Jm   = np.ones(Jtm.shape,     dtype=np_type) * -1
    J_dj = np.ones(J_up.shape[0], dtype=np_type) * def_val_dtgb
    # iterate all cells
    Jtm_it = np.nditer(Jtm, flags=['multi_index'])
    while not Jtm_it.finished:
        # if cell is a valid ID, find cell in list
        if Jtm_it[0] != -1:
            Jm[Jtm_it.multi_index] = J[np.flatnonzero(J_jt == Jtm_it[0])]
        Jtm_it.iternext()
    
    # create real representative index list for dummy cells iterating all
    # final indices
    for jj in range(1, J_up.shape[0]+1):
        jji = jj - 1
        # if cell is a dummy, find the connected real cell
        if J_type[jji] == str_dummy:
            # follow dummy cascade downwards until real cell and set index
            mmi = jji
            while J_type[mmi] == str_dummy:
                mm  = J_down[mmi]
                mmi = mm - 1
            J_dj[jji] = mm
    
    # calculate cell name and coordinates
    if print_out: print('...calculate coordinates...')
    # preallocate variable
    X     = []
    Y     = []
    # iterate final index
    for jj in range(1, J_down.shape[0]+1):
        jji = jj - 1
        # if jj is a dummy, insert X and Y coordinates using dummy list
        if J_type[jji] == str_dummy:
            # calculate coordinate indices
            xy = np.nonzero(Jm == J_dj[jji])
        # if it is a head water or routing cell, insert X and Y coordinates
        # using index array
        else:
            # calculate coordinate indices
            xy = np.nonzero(Jm == jj)
        # if jj is no dummy, insert X and Y coordinates
        X.append(xll + (xy[1][0]             + 1 - 0.5) * cellsz)
        Y.append(yll + (nrows - xy[0][0] - 1 + 0.5) * cellsz)
    
    # calculate upstream inflow catchment area of each routing cell
    # pre-allocate variable
    J_A = np.zeros(J.shape)
    # iterate all cells
    for jj_ii, jj in enumerate(J):
        # if it is a routing or the outflow cell, calculate area
        if J_type[jj_ii] == str_routing:
            J_A[jj_ii] = fa[Jm == jj] * ((cellsz / 1000)**2)
    
    # export model cell to point feature classes
    if print_out: print('...create model cell point feature classes...')
    # create pandas data frames
    structarr_tgb_in = list(zip(J_down, J_type, J_A, X, Y))
    df_mn = pd.DataFrame(structarr_tgb_in, index=J,
                         columns=[f_tgb_down, f_tgb_type, f_tgb_a, f_x, f_y])
    df_tgb_up = pd.DataFrame(J_up, index=J, columns=[f_tgb_up1, f_tgb_up2])
    # create real representative index list for dummy subcatchments
    ser_tgb_dtgb = real_repr_idx(df_mn, str_dummy, print_out=print_out)
    # create names of model subcatchments
    ser_nrflv = pd.Series(df_mn.shape[0] * '', index=df_mn.index, name=f_nrflv)
    for tgb, el_type in df_mn.loc[:, f_tgb_type].iteritems():
        ser_nrflv.at[jj] = '{0:s}{1:05d}'.format(el_type[0].upper(), tgb)
    # summarize DataFrames
    df_tgb = pd.concat([df_mn, ser_tgb_dtgb, ser_nrflv], axis=1)
    # summarize information for export
    ser_tgb = df_tgb.index.to_series(name=f_tgb)
    df_data_tgb_p = pd.concat(
            [ser_tgb, df_tgb.loc[:, [f_tgb_down, f_tgb_type,
                                     f_tgb_dtgb, f_tgb_a, f_x, f_y]]], axis=1)
    # create spatial reference object
    sr_obj = arcpy.Describe(path_fd_mr_e).spatialReference
    # export to point feature classes
    tc.tgb_to_points(df_data_tgb_p, sr_obj, path_gdb_out, name_tgb_p, 
                     geometry_fields=(f_x, f_y))
    tc.tgb_to_points(df_data_tgb_p, sr_obj, path_gdb_out, name_tgb_down_p,
                     geometry_fields=(f_x, f_y))
    
    # create model network polyline feature class
    if print_out: print('...create model network polyline feature class...')
    # import cell information
    arcpy.AddIndex_management(name_tgb_p, f_tgb_down, f_tgb_down,
                              'NON_UNIQUE', 'NON_ASCENDING')
    # delete non-relevant fields of downstream feature class
    arcpy.DeleteField_management(name_tgb_down_p, '{0}; {1}; {2}; {3}'.format(
            f_tgb_dtgb, f_tgb_down, f_tgb_type, f_tgb_a))
    # add coordinates to both feature classes
    arcpy.AddXY_management(name_tgb_p)
    arcpy.AddXY_management(name_tgb_down_p)
    # alter coordinate fields of downstream feature class
    f_p_xd = 'POINT_Xd'
    f_p_yd = 'POINT_Yd'
    arcpy.AlterField_management(name_tgb_down_p, f_p_x, f_p_xd, f_p_xd,
                                '', '4', 'NULLABLE', 'DO_NOT_CLEAR')
    arcpy.AlterField_management(name_tgb_down_p, f_p_y, f_p_yd, f_p_yd,
                                '', '4', 'NULLABLE', 'DO_NOT_CLEAR')
    # join information from downstream cells
    tgb_l_join = arcpy.management.AddJoin(name_tgb_p, f_tgb_down, name_tgb_down_p,
                                          f_tgb, 'KEEP_COMMON')
    # calculate line features
    if arcpy.Exists(path_mnw): arcpy.management.Delete(path_mnw)
    arcpy.XYToLine_management(tgb_l_join, path_mnw, f_p_x, f_p_y,
                              f_p_xd, f_p_yd, 'GEODESIC', f_tgb, sr_obj)
    # delete downstream neighbour model cell point feature class
    arcpy.Delete_management(name_tgb_down_p)
    
    return df_data_tgb_p, df_tgb_up
    
# %% function to preprocesses raster files, which are used for routing parameters.
def prepr_routing_rasters(path_dem_hr, path_fnw, path_ws_s, path_fa_mr, 
                          path_gdb_out, name_fa_hr='fa_hr',
                          name_dem_hr_ws='dem_hr_ws', name_fl_fnw_mr='fl_fnw_mr',
                          name_dem_max_mr='dem_max_mr', name_dem_min_mr='dem_min_mr',
                          initial=True, print_out=False):
    """
    Preprocesses raster files, which are used to calculate routing parameters.

    JM 2021

    Arguments:
    -----------
    path_dem_hr: str (e.g., 'c:\model_creation\dem_hr')
        path of the output watershed polygon feature class
    path_fnw: str (e.g., 'c:\model_creation\fnw')
        path of the flow network polyline feature class or shape file
    path_ws_s: str (e.g., 'c:\model_creation\ws_s')
        path of the model watershed polygon feature class
    path_fa_mr: str (e.g., 'c:\model_creation\fa_mr')
        path of the model resolution flow accumulation raster
    path_gdb_out: str (e.g., 'c:\model_creation.gdb')
        path of the output file geodatabase
    name_fa_hr: str (optional, default: 'fa_hr')
        name of the output extracted high resolution flow accumulation raster
    name_dem_hr_ws: str (optional, default: 'dem_hr_ws')
        name of the output extracted high resolution digital elevation raster
    name_fl_fnw_mr: str (optional, default: 'fl_fnw_mr')
        name of the output model resolution flow length at flow network location
    name_dem_max_mr: str (optional, default: 'dem_max_mr')
        name of the output model resolution maximum value of the high resolution DEM
    name_dem_min_mr: str (optional, default: 'dem_min_mr')
        name of the output model resolution minimum value of the high resolution DEM
    initial: boolean (optional, default: True)
        true if it is the first run and all steps have to be calculated from the scratch
    print_out: boolean (optional, default: False)
        true if workprogress shall be print to command line

    Returns:
    -----------
    Saves the following outputs:
        - extracted high resolution digital elevation raster (model domain)
        - model resolution flow length at flow network location (else: NaN)
        - model resolution maximum value of the high resolution elevation raster
        - model resolution minimum value of the high resolution elevation raster
    """
    # arcpy field names (ONLY CHANGE IF ARCPY FUNCTION CHANGES FORCE YOU!)
    f_oid       = 'OBJECTID'
    f_val       = 'Value'
    f_cellsz_x  = 'CELLSIZEX'
    method_mean = 'MEAN'
    method_max  = 'MAXIMUM'
    method_min  = 'MINIMUM'
    # paths for intermediates
    path_fnw_r      = path_gdb_out + 'fnw_r'
    path_fnw_mr     = path_gdb_out + 'fnw_mr'
    path_dem_hr_f   = path_gdb_out + 'dem_hr_f'
    path_fd_hr      = path_gdb_out + 'fd_hr'
    path_fl_hr      = path_gdb_out + 'fl_hr'
    path_fl_snfnw   = path_gdb_out + 'fl_snfnw'
    path_fl_snfa_mr = path_gdb_out + 'fl_snfa_mr'
    if initial: path_fl_aggr_mr = path_gdb_out + 'fl_aggr_mr'
    # paths for outputs
    path_dem_hr_ws = path_gdb_out + name_dem_hr_ws
    path_fa_hr     = path_gdb_out + name_fa_hr
    path_fl_fnw_mr = path_gdb_out + name_fl_fnw_mr
    if initial: 
        path_dem_max_mr = path_gdb_out + name_dem_max_mr
        path_dem_min_mr = path_gdb_out + name_dem_min_mr
    
    # set workspace
    arcpy.env.workspace = path_gdb_out
    # if it is the first calculation run, calculate high-resolution flow length
    if initial:
        if print_out: print('...calculate high-resolution flow length...')
        # save default environments
        default_env_snr = arcpy.env.snapRaster
        default_env_ext = arcpy.env.extent
        # set environments
        arcpy.env.extent     = 'MAXOF'
        arcpy.env.snapRaster = path_dem_hr
        # clip high resolution digital elevation raster to model domain
        if print_out: print('   step 1/7: clip high resolution DEM to model domain...')
        if arcpy.Exists(path_dem_hr_ws): arcpy.management.Delete(path_dem_hr_ws)
        dem_hr_ws      = arcpy.sa.ExtractByMask(path_dem_hr, path_ws_s)
        dem_hr_ws.save(path_dem_hr_ws)
        # fill corrected high resolution digital elevation raster
        if print_out: print('   step 2/7: fill clipped high resolution DEM...')
        if arcpy.Exists(path_dem_hr_f): arcpy.management.Delete(path_dem_hr_f)
        dem_hr_f      = arcpy.sa.Fill(path_dem_hr_ws, None)
        dem_hr_f.save(path_dem_hr_f)
        # calculate flow direction for filled digital elevation raster
        if print_out: print('   step 3/7: calculate high resolution flow direction...')
        if arcpy.Exists(path_fd_hr): arcpy.management.Delete(path_fd_hr)
        fd_hr      = arcpy.sa.FlowDirection(path_dem_hr_f, 'NORMAL', None, 'D8')
        fd_hr.save(path_fd_hr)
        # calculate flow accumulation
        if print_out: print('   step 4/7: calculate high resolution flow accumulation...')
        if arcpy.Exists(path_fa_hr): arcpy.management.Delete(path_fa_hr)
        fa_hr      = arcpy.sa.FlowAccumulation(path_fd_hr)
        fa_hr.save(path_fa_hr)
        # calculate flow length for flow direction
        if print_out: print('   step 5/7: calculate high resolution flow length...')
        if arcpy.Exists(path_fl_hr): arcpy.management.Delete(path_fl_hr)
        fl_hr      = arcpy.sa.FlowLength(path_fd_hr, 'DOWNSTREAM', None)
        fl_hr.save(path_fl_hr)
        # convert flow network polyline feature class to high resolution raster
        if print_out: print(('   step 6/7: convert flow network polyline feature '
                             'class to high resolution raster...'))
        if arcpy.Exists(path_fnw_r): arcpy.management.Delete(path_fnw_r)
        arcpy.conversion.PolylineToRaster(path_fnw, 'OBJECTID', path_fnw_r,
                                          'MAXIMUM_LENGTH', 'NONE', path_dem_hr_ws)
        # set flow length to nan if flow network raster is nan
        if print_out: print(('   step 7/7: set flow length to nan if flow network '
                             'raster is nan...'))
        if arcpy.Exists(path_fl_snfnw): arcpy.management.Delete(path_fl_snfnw)
        setn_expr = '{0} IS NULL'.format(f_val)
        fl_snfnw      = arcpy.ia.SetNull(path_fnw_r, path_fl_hr, setn_expr)
        fl_snfnw.save(path_fl_snfnw)
        # reset environments
        arcpy.env.snapRaster = default_env_snr
        arcpy.env.extent     = default_env_ext
    
    # Aggregate flow length to model resolution
    if print_out: print('...aggregate flow length to model resolution...')
    # save default environments
    default_env_snr      = arcpy.env.snapRaster
    default_env_ext      = arcpy.env.extent
    default_env_mask     = arcpy.env.mask
    # set environments
    arcpy.env.snapRaster = path_fa_mr
    arcpy.env.extent     = path_fa_mr
    arcpy.env.mask       = path_fa_mr
    # get high resolution and model resolution cell size
    cell_sz_x_obj    = arcpy.GetRasterProperties_management(path_dem_hr_ws, f_cellsz_x)
    cell_sz_x        = np.int32(cell_sz_x_obj.getOutput(0))
    cellsz_obj = arcpy.GetRasterProperties_management(path_fa_mr, f_cellsz_x)
    cellsz     = np.int32(cellsz_obj.getOutput(0))
    # aggregate flow length to final cell size
    if initial:
        fl_aggr_mr = arcpy.sa.Aggregate(
                path_fl_snfnw, str(np.int32(cellsz/cell_sz_x)),
                method_mean, 'EXPAND', 'DATA')
        fl_aggr_mr.save(path_fl_aggr_mr)
    # set aggregated flow length at flow accumulation areas < 0.1 km² to nan
    expr_sql   = '{0:s} < {1:.0f}'.format(f_val, 1000/cellsz)
    fl_snfa_mr = arcpy.ia.SetNull(path_fa_mr, path_fl_aggr_mr, expr_sql)
    fl_snfa_mr.save(path_fl_snfa_mr)
    # convert polylines to raster in model grid size
    arcpy.conversion.PolylineToRaster(path_fnw, f_oid, path_fnw_mr,
                                      'MAXIMUM_LENGTH', 'NONE', path_fa_mr)
    # set aggregated flow length to nan if aggregated flow network is nan as well
    if arcpy.Exists(path_fl_fnw_mr): arcpy.management.Delete(path_fl_fnw_mr)
    fl_fnw_mr      = arcpy.ia.SetNull(arcpy.ia.IsNull(path_fnw_mr), path_fl_snfa_mr)
    fl_fnw_mr.save(path_fl_fnw_mr)
    
    # Aggregate high-resolution DEM to model resolution extracting min and max values
    if initial:
        if print_out: print(('...calculate min and max high resolution DEM values '
                             'in model resolution...'))
        if arcpy.Exists(path_dem_max_mr): arcpy.management.Delete(path_dem_max_mr)
        if arcpy.Exists(path_dem_min_mr): arcpy.management.Delete(path_dem_min_mr)
        dem_max_mr = arcpy.sa.Aggregate(path_dem_hr_ws, str(cellsz),
                                        method_max, 'EXPAND', 'DATA')
        dem_min_mr = arcpy.sa.Aggregate(path_dem_hr_ws, str(cellsz),
                                        method_min, 'EXPAND', 'DATA')
        dem_max_mr.save(path_dem_max_mr)
        dem_min_mr.save(path_dem_min_mr)
    # reset environments
    arcpy.env.snapRaster = default_env_snr
    arcpy.env.extent     = default_env_ext
    arcpy.env.mask       = default_env_mask


# %% function to create point feature class indicating elements, where no 
# high-resolution flow length shall be calculated
def create_fl_ind_point(sr_obj,
                        path_gdb_out, name_no_fnw_fl='no_fnw_fl',
                        print_out=False):
    """
    Creates a point feature class in the defined file geodatabase to be filled
    by the user with points. These points indicate cells, for which no high
    resolution flow length shall be calculated, but the model resolution is used
    instead. The point feature class has neither Z- nor M-values.

    JM 2021

    Arguments:
    -----------
    sr_obj: arcpy.SpatialReferenceObject
        arcpy.Object containing the spatial reference of the final feature class
    path_gdb_out: str
        path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
    name_no_fnw_fl: str (optional)
        name of the output indication point feature class (e.g., 'no_fnw_fl')
    print_out: boolean
        true if workprogress shall be print to command line

    Returns:
    -----------
    Saves the output pour point feature class
    """
    if print_out: print('...create indication point feature class...')
    # set path for output
    path_no_fnw_fl = path_gdb_out + name_no_fnw_fl
    # prepare indication point feature class
    if arcpy.Exists(path_no_fnw_fl): arcpy.management.Delete(path_no_fnw_fl)
    arcpy.CreateFeatureclass_management(path_gdb_out, name_no_fnw_fl, 'POINT', '',
                                        'DISABLED', 'DISABLED', sr_obj, '', '0',
                                        '0', '0', '')

# %% Create polygon feature class representing the model elements
def create_element_polyg(path_tgb_p, path_sn_raster, path_gdb_out,
                         name_tgb_s='tgb_s', print_out=False):
    """
    Creates a polygon feature class in the defined file geodatabase, which includes
    all values of the input point feature class and represents the model element
    raster structure. The feature class only includes model elements, which are no
    dummy elements and covers the whole model domain.

    JM 2021

    Arguments:
    -----------
    path_tgb_p: str
        path of the flow network feature class or shape file
        (e.g., 'c:\model_creation.gdb\tgb_p')
    path_sn_raster: str
        path of the raster, which represents the model raster
    path_gdb_out: str
        path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
    name_tgb_s: str (optional, default: 'tgb_s')
        name of the output model element polygon feature class
    print_out: boolean
        true if workprogress shall be print to command line

    Returns:
    -----------
    Saves a polygon feature class representing the model elements
    """
    # define internal variables
    def_val_dtgb = -1
    # define internal field names
    f_tgb      = 'tgb'
    f_tgb_dtgb = 'tgb_dtgb'
    # arcpy field names (ONLY CHANGE IF ARCPY FUNCTION CHANGES FORCE YOU!)
    f_val      = 'VALUE'
    f_cellsz_x = 'CELLSIZEX'
    f_gridcode = 'gridcode'
    # define paths of intermediates in working geodatabase
    name_tgb_p_nodum = 'tgb_p_nodum'
    name_tgb_p_sel   = 'tgb_p_sel'
    name_tgb_r       = 'tgb_r'
    # set workspace
    arcpy.env.workspace = path_gdb_out
    # Calculate model element polygons in raster structure
    if print_out: print('...Calculate model element polygons in raster structure...')
    # save original environment settings
    default_env_snr = arcpy.env.snapRaster
    # select elements which are not dummys and copy features to a new layer
    sel_expr = '{0:s} = {1:d}'.format(f_tgb_dtgb, def_val_dtgb)
    name_tgb_p_sel = arcpy.management.SelectLayerByAttribute(
            path_tgb_p, 'NEW_SELECTION', sel_expr, '')
    arcpy.CopyFeatures_management(name_tgb_p_sel, name_tgb_p_nodum, '', '', '', '')
    arcpy.management.SelectLayerByAttribute(path_tgb_p, 'CLEAR_SELECTION', '', None)
    # set environment
    arcpy.env.snapRaster = path_sn_raster
    # get model cell size
    cellsz_obj = arcpy.GetRasterProperties_management(path_sn_raster, f_cellsz_x)
    cellsz     = np.int32(cellsz_obj.getOutput(0))
    # create elment features from point layer converting to raster
    arcpy.PointToRaster_conversion(name_tgb_p_nodum, f_tgb, name_tgb_r,
                                   'MOST_FREQUENT', 'NONE', str(cellsz))
    arcpy.RasterToPolygon_conversion(name_tgb_r, name_tgb_s, 'NO_SIMPLIFY',
                                     f_val, 'SINGLE_OUTER_PART', '')
    arcpy.AlterField_management(name_tgb_s, f_gridcode, f_tgb, f_tgb)
    # restore environment
    arcpy.env.snapRaster = default_env_snr

# %% summyrize GIS data for runoff concentration and routing parameter calculation
def summar_gisdata_for_roandrout(
        path_tgb_p, path_dem_max_mr, path_dem_min_mr, 
        path_fl_mr, path_fl_fnw_mr, path_no_fnw_fl, 
        path_gdb_out, name_tgb_par_p='tgb_par_p',
        field_dem_max_mr='dem_max_mr', field_dem_min_mr='dem_min_mr', 
        field_fl_fnw_mean_mr='fl_fnw_mean_mr', field_fl_mr='fl_mr',
        print_out=False):
    """
    Creates a point feature class in the defined file geodatabase, which includes
    values of the maximum and minimum elevation as well as the flow network and
    the model resolution flow length within each model element. 

    JM 2021

    Arguments:
    -----------
    path_tgb_p: str
        path of the flow network feature class or shape file
        (e.g., 'c:\model_creation.gdb\tgb_p')
    path_dem_max_mr: str
        path of the flow network feature class or shape file
        (e.g., 'c:\model_creation.gdb\dem_max_mr')
    path_dem_min_mr: str
        path of the output file geodatabase
        (e.g., 'c:\model_creation.gdb\dem_min_mr')
    path_fl_mr: str
        path of the output file geodatabase
        (e.g., 'c:\model_creation.gdb\fl_mr')
    path_fl_fnw_mr: str
        path of the flow network feature class or shape file
        (e.g., 'c:\model_creation.gdb\fl_fnw_mr')
    path_no_fnw_fl: str
        path of the output file geodatabase
        (e.g., 'c:\model_creation.gdb\no_fnw_fl')
    path_gdb_out: str
        path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
    name_tgb_par_p: str (optional, default: 'tgb_par_p')
        name of the output model element point feature class with extracted
        parameters
    field_dem_max_mr: str (optional, default: 'dem_max_mr')
        name of the field in name_tgb_par_p containing max elevation value
    field_dem_min_mr: str (optional, default: 'dem_min_mr')
        name of the field in name_tgb_par_p containing min elevation value
    field_fl_fnw_mean_mr: str (optional, default: 'fl_fnw_mean_mr')
        name of the field in name_tgb_par_p containing flow network flow length
    field_fl_mr: str (optional, default: 'fl_mr')
        name of the field in name_tgb_par_p containing model resolution flow length
    print_out: boolean
        true if workprogress shall be print to command line

    Returns:
    -----------
    Saves model element point feature class with extracted parameters:
        - minimum elevation
        - maximum elevation
        - model resolution flow length
        - flow network flow length
    """
    # define internal variables
    f_tgb = 'tgb'
    # set workspace
    arcpy.env.workspace = path_gdb_out
    # define paths of intermediates in working geodatabase
    name_fl_fnw_mr_corr = 'fl_fnw_mr_corr'
    name_no_fnw_fl_r    = 'no_fnwfl_r'
    # save original environment settings
    default_env_snr = arcpy.env.snapRaster
    default_env_ext = arcpy.env.extent
    
    # If field fl_mr (model resolution flow length) does not exist,
    # add field while extracting flow length values
    if not arcpy.ListFields(path_tgb_p, field_fl_mr):
        if print_out: print('...extract flow length values...')
        arcpy.gp.ExtractMultiValuesToPoints_sa(
                path_tgb_p, path_fl_mr + ' ' + field_fl_mr, 'NONE')
    
    # If there are any flow length correction points
    # remove values of fl_mr at TGBs marked with a feature point in no_fnw_fl
    if arcpy.management.GetCount(path_no_fnw_fl)[0] != '0':
        if print_out: print('...correct marked flow length values...')
        # set environments
        arcpy.env.extent     = 'MAXOF'
        arcpy.env.snapRaster = path_fl_mr
        # convert correction points to raster and remove flow network flow length values
        arcpy.PointToRaster_conversion(path_no_fnw_fl, 'OBJECTID', name_no_fnw_fl_r,
                                       'MOST_FREQUENT', 'NONE', path_fl_mr)
        fl_fnw_mr_corr = arcpy.ia.Con(arcpy.ia.IsNull(name_no_fnw_fl_r), path_fl_fnw_mr)
        fl_fnw_mr_corr.save(name_fl_fnw_mr_corr)
        # restore environments
        arcpy.env.extent     = default_env_ext
        arcpy.env.snapRaster = default_env_snr
        
    # Extract min and max elevation and flow length values to model point features
    if print_out: print('...extract raster values to model point features...')
    # copy model element point features to a new feature class
    arcpy.management.CopyFeatures(path_tgb_p, name_tgb_par_p)
    # if there are any flow length correction points, add information from corrected
    if arcpy.management.GetCount(path_no_fnw_fl)[0] != '0':
        arcpy.sa.ExtractMultiValuesToPoints(name_tgb_par_p, [
                [path_dem_max_mr, field_dem_max_mr],
                [path_dem_min_mr, field_dem_min_mr],
                [name_fl_fnw_mr_corr, field_fl_fnw_mean_mr]], 'NONE')
    # else use original files
    else:
        arcpy.sa.ExtractMultiValuesToPoints(name_tgb_par_p, [
                [path_dem_max_mr, field_dem_max_mr],
                [path_dem_min_mr, field_dem_min_mr],
                [path_fl_fnw_mr, field_fl_fnw_mean_mr]], 'NONE')
    # delete identical (Workaround for Bug in ExtractMultiValuesToPoints)
    arcpy.management.DeleteIdentical(name_tgb_par_p, f_tgb, None, 0)
    
# %% calculate parameters for tgb.dat
def calc_roandrout_params(cellsz, q_spec_ch, name_tgb_par_p,
        field_dem_max_mr='dem_max_mr', field_dem_min_mr='dem_min_mr', 
        field_fl_mr='fl_mr', field_fl_fnw_mean_mr='fl_fnw_mean_mr',
        def_fl_upper_lim=np.inf, def_fl_strct_mism=2, def_sl_min=0.0001,
        def_sl_excl_quant=None, def_zmin_rout_fac=0.5, def_zmax_fac=1,
        ser_q_in_corr=None, ch_est_method='combined', def_bx=0, def_bbx_fac=1,
        def_bnm=1.5, def_bnx=100, def_bnvrx=4, def_skm=30, def_skx=20, 
        print_out=False):
    
    """
    Creates a point feature class in the defined file geodatabase, which includes
    values of the maximum and minimum elevation as well as the flow network and
    the model resolution flow length within each model element. 

    JM 2021

    Arguments:
    -----------
    cellsz: integer
        edge length of the model elements in [m] (e.g., 100)
    q_spec_ch: float
        channel forming specific flood discharge value [m3s-1km-2] (e.g., 0.21)
    name_tgb_par_p: str
        path of the input model element point feature class with following
        parameters for each element except dummy elements:
            - maximum elevation value
            - minimum elevation value
            - channel model resolution flow length
            - channel flow network flow length
        (e.g., 'c:\model_creation.gdb\tgb_p_fl')
    field_dem_max_mr: str (optional, default: 'dem_max_mr')
        name of the field in name_tgb_par_p containing max elevation value
    field_dem_min_mr: str (optional, default: 'dem_min_mr')
        name of the field in name_tgb_par_p containing min elevation value
    field_fl_mr: str (optional, default: 'fl_mr')
        name of the field in name_tgb_par_p containing model resolution flow length
    field_fl_fnw_mean_mr: str (optional, default: 'fl_fnw_mean_mr')
        name of the field in name_tgb_par_p containing flow network flow length
    def_sl_min: float (optional, default: 0.0001)
        minimum channel slope value to be maintained due to LARSIM-internal
        restrictions
    def_fl_strct_mism: int (optional, default: 2)
        default flow length for structural mismatch and negative transition
        deviations [m]. attention: 1 [m] is interpreted by LARSIM as 'no routing'!
    def_fl_upper_lim: int (optional, default: inf)
        upper threshold for realistic flow length [m]
    def_sl_excl_quant: float (optional, default: None)
        quantile of slope values to be set constant to quantile value
        (e.g., 0.999 sets the upper 0.1% of the slope values to the
        0.1% quantile value)
    def_zmin_rout_fac: float (optional, default: 0.5)
        Factor to vary the lower elevation of runoff concentration between
        the minimum (0) and maximum (1) channel elevation of the element.
        By default, the factor is set to the average elevation (0.5) [-] 
    def_zmax_fac: float (optional, default: 1)
        Factor to vary the upper elevation of runoff concentration between
        the minimum (0) and maximum (1) elevation of the element. By default,
    ser_q_in_corr: pandas.Series
        Series of channel-forming inflow (e.g., HQ2) at the corresponding 
        model element ID in the serie's index. 
        (e.g., pd.Series(np.array([2.8, 5.3]), index=[23, 359], name='q_in'))
    ch_est_method: string (optional, default: 'combined')
        String defining channel estimation function. Possible values: 
        - 'Allen': Allen et al. (1994)
        - 'Krauter': Krauter (2006)
        - 'combined': Allen et al.(1994) for small and Krauter (2006) for
                      large areas
    def_bx: float (optional, default: 0)
        Float defining the flat foreland width left and right [m]
    def_bbx_fac: float (optional, default: 1)
        Float factor defining the slopy foreland width left and right,
        which is calculated multiplying the channel width with this factor [-]
    def_bnm: float (optional, default: 1.5 = 67%)
        Float defining the channel embankment slope left and right [mL/mZ]
    def_bnx: float (optional, default: 100 = nearly flat foreland)
        Float defining the slopy foreland slope left and right [mL/mZ]
    def_bnvrx: float (optional, default: 4 = 25%)
        Float defining the outer foreland slope left and right [mL/mZ]
    def_skm: float (optional, default: 30 = natural channel, vegetated river bank)
        Float defining the Strickler roughness values in the channel [m1/3s-1]
    def_skx: float (optional, default: 20 = uneven vegetated foreland)
        Float defining the Strickler roughness values of the left and right
        foreland [m1/3s-1]
    print_out: boolean (optional, default: False)
        true if workprogress shall be print to command line

    Returns:
    -----------
    df_data_tgbdat: pandas.DataFrame
        DataFrame of all parameters, which are needed in the resulting file.
        The DataFrame includes the model element ID as index and the following
        columns:
        - 'TGB': element ID number (int)
        - 'NRVLF': element name (str)
        - 'FT': element area (float)
        - 'HUT': lower elevation of runoff concentration [m]
        - 'HOT': upper elevation of runoff concentration [m]
        - 'TAL': maximum flow length for runoff concentration [km]
        - 'X': x-coordinate of element center [m]
        - 'Y': y-coordinate of element center [m]
        - 'KMU': lower stationing of routing [m]
        - 'KMO': upper stationing of routing [m]
        - 'GEF': channel slope for routing [m]
        - 'HM': channel depth [m]
        - 'BM': channel width [m]
        - 'BL': flat foreland width left [m]
        - 'BR': flat foreland width right [m]
        - 'BBL': slopy foreland width left [m]
        - 'BBR': slopy foreland width right [m]
        - 'BNM': channel embankment slope left and right [mL/mZ]
        - 'BNL': slopy foreland slope left [mL/mZ]
        - 'BNR': slopy foreland slope right [mL/mZ]
        - 'BNVRL': outer foreland slope left [mL/mZ]
        - 'BNVRR': outer foreland slope right [mL/mZ]
        - 'SKM': Strickler roughnes values in the channel [m1/3s-1]
        - 'SKL': Strickler roughnes values at the left foreland [m1/3s-1]
        - 'SKR': Strickler roughnes values at the right foreland [m1/3s-1]
    ser_tgb_down_nd: pandas.Series
        Series of corresponding downstream model element indices ignoring dummy
        elements. Model outlet remains -1 and dummy elements are represented as 0.
    ser_ft: pandas.Series
        Series of corresponding model element areas. Dummy elements have an
        area of 0. [km²]
    ser_area_outfl: pandas.Series
        Series of corresponding model element inflow catchment areas. 
        Dummy elements have an area of 0. [km²]
    ser_ch_form_q: pandas.Series
        Series of elements' channel-forming discharge at the corresponding
        model element ID in the serie's index. 
    """    

    # %% Redistribute flow length values at confluence points
    def redistr_flowl_at_conflp(ser_fl, ser_tgb_down_nd, ser_tgb_up_nd,
                                ser_tgb_type_headw, ser_tgb_type_dummy):
        """
        This function redistributes flow length values at cofluence points. 
        Remember: there will result multipliers of 1 and sqrt(2) with the model 
        resolution as flow length values from D8 flow length calculation. The LARSIM
        convention assumes the confluence point of cells upstream of the routing
        element. Therefore, the resulting discrepancies at confluence points have to
        be balanced in upstream routing elements.
    
        JM 2021
    
        Arguments:
        -----------
        ser_fl: pandas.Series
            Series of model element raster flow length corresponding to the serie's
            ascending index. The flow length is calculated using the D8-flow direction
            based on the model resolution digital elevation raster and using the 
            'DOWNSTREAM' option (outlet = 0). It may be clipped from a larger raster,
            whereby the outlet is not zero anymore. 
            (e.g., pd.Series([300, 341, 200, 100], index=[1, 2, 3, 4], name='ser_fl'))
        ser_tgb_down_nd: pandas.Series
            Series of corresponding downstream model element indices ignoring dummy
            elements. Model outlet remains -1 and dummy elements are represented as 0.
        ser_tgb_up_nd: pandas.Series
            Series of corresponding upstream model element indices ignoring dummy
            elements. These are represented as empty array (e.g., []).
        ser_tgb_type_headw: pandas.Series
            Boolean Series, which identifies the headwater cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[1, 1, 0, 0], index=[1, 2, 3, 4], name='headwater',
                             dtype='bool'))
        ser_tgb_type_dummy: pandas.Series
            Boolean Series, which identifies the dummy cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[0, 0, 0, 0], index=[1, 2, 3, 4], name='dummy',
                             dtype='bool'))
    
        Returns:
        -----------
        df_fl: pandas.DataFrame
            DataFrame of corresponding model resolution flow length values. The DataFrame
            includes the following columns:
                - downstream share of flow length within cell ('down')
                - downstream share confluence correction value ('corr_conf_down')
                - corrected downstream share of flow length within cell ('corr_down')
                - upstream share of flow length within cell ('up')
                - corrected upstream share of flow length within cell ('corr_up')
        """
        # define internal string variables
        f_up             = 'up'
        f_down           = 'down'
        f_corr_up        = 'corr_up'
        f_corr_down      = 'corr_down'
        f_corr_conf_down = 'corr_conf_down'
        # pre-allocate variable
        df_fl = pd.DataFrame(np.zeros((ser_fl.shape[0], 5)) * np.nan,
                             index=ser_fl.index,
                             columns=[f_down, f_corr_conf_down, f_corr_down,
                                      f_up, f_corr_up])
        # copy model resolution flow length (GIS raster calculation)
        # (dummy cells are nan, outflow not)
        ser_fl.at[ser_tgb_type_dummy] = np.nan
        df_fl.at[ser_tgb_type_dummy, :] = np.nan
        # iterate elements to calculate flow length to downstream cell
        # (only head water and routing cells)
        for tgb, fl_sum_tgb in ser_fl.iteritems():
            # if it is a head water or routing cell and not the outflow:
            if not ser_tgb_type_dummy.at[tgb] and tgb != np.max(df_fl.index):
                # get flow length of downstream cell
                fl_sum_down = ser_fl.loc[ser_tgb_down_nd.at[tgb]]
                # calculate flow length difference between recent and downstream cell
                df_fl.at[tgb, f_down] = (fl_sum_tgb - fl_sum_down) / 2
        # iterate elements to calculate flow length to upstream cells and correct it
        for tgb, fl_sum_tgb in ser_fl.iteritems():
            # if it is a head water cell set upstream flow length to zero
            if ser_tgb_type_headw.at[tgb]:
                df_fl.at[tgb, f_up] = 0
            # if it is a routing cell allocate mean residuals to upstream cells
            elif not ser_tgb_type_dummy.at[tgb]:
                # get values of upstream cells
                fl_sum_up = ser_fl.loc[ser_tgb_up_nd.at[tgb]]
                # calculate mean of differences between recent and upstream cells
                fl_dif_up = np.nanmean(fl_sum_up - fl_sum_tgb) / 2
                df_fl.at[tgb, f_up] = fl_dif_up
                # calculate mean downstream residuals and allocate it to upstream cells
                fl_dif_up_rest = (fl_sum_up - fl_sum_tgb) / 2 - fl_dif_up
                df_fl.loc[fl_dif_up_rest.index, f_corr_conf_down] = fl_dif_up_rest
        # calculate sums of flow length shares
        df_fl.loc[:, f_corr_down] = np.sum(
                df_fl.loc[:, [f_down, f_corr_conf_down]], axis=1)
        df_fl.loc[:, f_corr_up] = df_fl.loc[:, f_up].values
        return df_fl
    
    # %% Redistribute flow network flow length values at confluence points
    def redistr_flowl_polyl_at_conflp(ser_fl_fnw, ser_tgb_down_nd, ser_tgb_up_nd,
                                      ser_tgb_type_headw, ser_tgb_type_dummy, cellsz):
        """
        This function redistributes the model resolution flow length values calculated
        based on existing flow path polyline features. 
        Remember: The LARSIM convention assumes the confluence point of cells
        upstream of the routing element. Therefore, the resulting discrepancies at
        confluence points have to be balanced in upstream routing elements.
        Furthermore, the flow network balances might get negative with unavoidable
        influences of neighbouring flow network elements. This will be retained by
        setting discrepancies to a symbolic value of 1 to prevent LARSIM assuming
        a dummy cell. As it stays unclear, where the influencing flow network element
        belongs to, the (rather small) discrepancy has to stay unbalanced upstream.
        Additionally, to prevent instabilities in the water routing calculation, a
        correction and redistribution of very small flow lengths is introduced. If
        the flow length is smaller than 10% of the model's cell size, the difference
        to the actual flow length at the recent cell is redistributed from upstream
        cells to the recent one.
    
        JM 2021
    
        Arguments:
        -----------
        ser_fl_fnw: pandas.Series
            Series of model element polyline flow length corresponding to the serie's
            ascending index. The flow length is calculated using the accumulative
            lengths of polyline elements intersected with model raster polygons.
            The outlet is the minimum value, but has not to be zero. 
            (e.g., pd.Series([308.4, 341.0, 204.5, 133.8], index=[1, 2, 3, 4],
                             name='ser_fl_fnw'))
        ser_tgb_down_nd: pandas.Series
            Series of corresponding downstream model element indices ignoring dummy
            elements. Model outlet remains -1 and dummy elements are represented as 0.
        ser_tgb_up_nd: pandas.Series
            Series of corresponding upstream model element indices ignoring dummy
            elements. These are represented as empty array (e.g., []).
        ser_tgb_type_headw: pandas.Series
            Boolean Series, which identifies the headwater cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[1, 1, 0, 0], index=[1, 2, 3, 4], name='headwater',
                             dtype='bool'))
        ser_tgb_type_dummy: pandas.Series
            Boolean Series, which identifies the dummy cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[0, 0, 0, 0], index=[1, 2, 3, 4], name='dummy',
                             dtype='bool'))
        cellsz: int
            Integer, which defines the model element edge length in [m]
    
        Returns:
        -----------
        df_fl_fnw: pandas.DataFrame
            DataFrame of corresponding model resolution flow length values.
            The DataFrame includes the following columns:
                - original downstream share of flow length within cell ('down')
                - downstream correction value of confluences ('corr_conf_down')
                - downstream correction value of redistribution ('corr_red_down')
                - corrected downstream share of flow length within cell ('corr_down')
                - upstream share of flow length within cell ('up')
                - upstream correction value of redistribution ('corr_red_up')
                - corrected upstream share of flow length within cell ('corr_up')
        """
        # define internal string variables
        f_up             = 'up'
        f_down           = 'down'
        f_corr_up        = 'corr_up'
        f_corr_down      = 'corr_down'
        f_corr_conf_down = 'corr_conf_down'
        f_corr_red_up    = 'corr_red_up'
        f_corr_red_down  = 'corr_red_down'
        # pre-allocate variable
        df_fl_fnw = pd.DataFrame(np.zeros((ser_fl_fnw.shape[0], 7))*np.nan,
                                 index=ser_fl_fnw.index,
                                 columns=[f_down, f_corr_conf_down,
                                          f_corr_red_down, f_corr_down,
                                          f_up, f_corr_red_up, f_corr_up])
        # first column = high resolution flow length (GIS raster calculation)
        # (dummy cells are nan, outflow not)
        ser_fl_fnw.at[ser_tgb_type_dummy] = np.nan
        df_fl_fnw.at[ser_tgb_type_dummy, :] = np.nan
        # calculate flow distances
        for tgb, fl_sum in ser_fl_fnw.iteritems():
            # if high resolution flow length is not nan...
            if not np.isnan(fl_sum):
                # if it is a head water cell only calculate downstream part
                if ser_tgb_type_headw.at[tgb]:
                    # find downstream cell and get flow length
                    fl_down = ser_fl_fnw.loc[ser_tgb_down_nd.at[tgb]]
                    # calculate flow length difference between recent and
                    # downstream cell
                    df_fl_fnw.at[tgb, f_down] = (fl_sum - fl_down) / 2
                    # set difference between recent and upstream cell to zero
                    df_fl_fnw.at[tgb, f_up] = 0
                # if it is a routing cell...
                elif not ser_tgb_type_dummy.at[tgb]:
                    # if it is not outflow
                    if tgb != np.max(ser_fl_fnw.index):
                        # find downstream cell and get flow length
                        fl_down = ser_fl_fnw.loc[ser_tgb_down_nd.loc[tgb]]
                        # downstream value is difference between recent and 
                        # downstream cell or 1 [m] if it would be smaller
                        df_fl_fnw.at[tgb, f_down] \
                            = np.max([(fl_sum - fl_down) / 2, 1])
                    else:
                        # downstream difference is 0
                        df_fl_fnw.at[tgb, f_down] = 0
                    # find upstream cells and get flow lengths
                    jjnd_up = ser_tgb_up_nd.at[tgb]
                    # calculate flow length difference between recent
                    # and upstream cells
                    fl_dif_up = (ser_fl_fnw.loc[jjnd_up] - fl_sum) / 2
                    # correct negative upstream difference values and protocol
                    fl_dif_up_ii = np.logical_and(np.isnan(fl_dif_up),
                                                  fl_dif_up < 0)
                    fl_dif_up[fl_dif_up_ii] = 1
                    # calculate mean of difference between recent
                    # and upstream cells
                    if np.any(~np.isnan(fl_dif_up)):
                        fl_difmean_up = np.nanmean(fl_dif_up)
                    else:
                        fl_difmean_up = np.nan
                    df_fl_fnw.at[tgb, f_up] = fl_difmean_up
                    # calculate residual from mean calculation
                    df_fl_fnw.at[jjnd_up, f_corr_conf_down] \
                        = fl_dif_up - fl_difmean_up
        # iterate cells in reversed calculation order from outflow to most
        # upstream point and redistribute very small network values
        for tgb in reversed(ser_fl_fnw.index):
            # if high resolution flow length is not nan and it is a routing cell...
            fl_sum = ser_fl_fnw[tgb]
            if not np.isnan(fl_sum) \
                and not (ser_tgb_type_headw.at[tgb] and ser_tgb_type_dummy.at[tgb]):
                # add downstream, upstream and remaining flow length part of
                # recent element
                fl_fnw = np.nansum(
                        df_fl_fnw.loc[tgb, [f_down, f_corr_conf_down, f_up]])
                # if the flow length is smaller than 10% of the cell size...
                if fl_fnw < cellsz / 10:
                    # allocate the difference to 10% of cell size to the
                    # recent element
                    fl_fnw_dif_corr = cellsz / 10 - fl_fnw
                    df_fl_fnw.at[tgb, f_corr_red_up] = fl_fnw_dif_corr
                    # redistribute correction length to upstream cells
                    df_fl_fnw.at[ser_tgb_up_nd.at[tgb], f_corr_red_down] \
                        = - fl_fnw_dif_corr
        # calculate sums of flow length shares
        df_fl_fnw.at[:, f_corr_down] = np.sum(
                df_fl_fnw.loc[:, [f_down, f_corr_conf_down, f_corr_red_down]], axis=1)
        df_fl_fnw.at[:, f_corr_up] = np.sum(
                df_fl_fnw.loc[:, [f_up, f_corr_red_up]], axis=1)
        return df_fl_fnw
    
    # %% Merge flow length from model resolution raster and flow network polylines
    def merge_fnw_and_mr_fl(df_fl_mr, df_fl_fnw, ser_j_down, ser_tgb_down_nd,
                            ser_tgb_type_headw, ser_tgb_type_dummy,
                            def_fl_upper_lim=np.inf, def_fl_strct_mism=2):
        """
        This function merges both model resolution flow length sources (1)
        calculated based on existing flow path polyline features and (2) using
        the D8-flow direction based on the model resolution digital elevation 
        raster and using the 'DOWNSTREAM' option (outlet = 0). 
        The flow length calculated from flow network polylines and model
        resolution are potentially referenced to a different outflow point as
        the extent of the DEM is different. The extent of the flow network
        usually is larger, as the calculation of the model domain is based on
        the underlying high-resolution DEM. Therefore, flow lenght references
        have to be reset at the outflow point of the model. Consequently, the
        flow network and model resolution flow length values are merged.
    
        JM 2021
    
        Arguments:
        -----------
        df_fl_mr: pandas.DataFrame
            DataFrame of corresponding model resolution flow length values. The 
            DataFrame includes the model element ID as index and the following columns:
                - downstream share of flow length within cell ('down')
                - corrected downstream share of flow length within cell ('corr_down')
                - corrected upstream share of flow length within cell ('corr_up')
        df_fl_fnw: pandas.DataFrame
            DataFrame of corresponding model resolution flow length values. The 
            DataFrame includes the model element ID as index and the following columns:
                - corrected downstream share of flow length within cell ('corr_down')
                - corrected upstream share of flow length within cell ('corr_up')
        ser_j_down: pandas.Series
            Series of corresponding downstream model element indices.
            Model outlet remains -1 and dummy elements are represented as 0.
        ser_tgb_down_nd: pandas.Series
            Series of corresponding downstream model element indices ignoring dummy
            elements. Model outlet remains -1 and dummy elements are represented as 0.
        ser_tgb_up_nd: pandas.Series
            Series of corresponding upstream model element indices ignoring dummy
            elements. These are represented as empty array (e.g., []).
        ser_tgb_type_headw: pandas.Series
            Boolean Series, which identifies the headwater cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[1, 1, 0, 0], index=[1, 2, 3, 4], name='headwater',
                             dtype='bool'))
        ser_tgb_type_dummy: pandas.Series
            Boolean Series, which identifies the dummy cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[0, 0, 0, 0], index=[1, 2, 3, 4], name='dummy',
                             dtype='bool'))
        def_fl_upper_lim: int (optional, default: inf)
            upper threshold for realistic flow length [m]
        def_fl_strct_mism: int (optional, default: 2)
            default flow length for structural mismatch and negative transition
            deviations [m]. attention: 1 [m] is interpreted by LARSIM as 'no routing'!
    
        Returns:
        -----------
        df_fl: pandas.DataFrame
            DataFrame of corresponding model resolution flow length values. 
            The DataFrame includes the model element ID as index and the following
            columns:
                - accumulative flow length at lower boundary of cell ('lower')
                - flow length value of cell ('length')
                - accumulative flow length at upper boundary of cell ('upper')
        """
        # define internal string variables
        f_up        = 'up'
        f_down      = 'down'
        f_corr_up   = 'corr_up'
        f_corr_down = 'corr_down'
        f_lower     = 'lower'
        f_upper     = 'upper'
        f_length    = 'length'
        # pre-allocate DataFrame for indentification keys
        df_fl_keys = pd.DataFrame(np.zeros((df_fl_mr.shape[0], 2))*np.nan,
                               index=df_fl_mr.index, columns=[f_down, f_up])
        # pre-allocate DataFrame for flow length values
        df_fl = pd.DataFrame(np.zeros((df_fl_mr.shape[0], 3))*np.nan,
                               index=df_fl_mr.index,
                               columns=[f_lower, f_length, f_upper])
        # calculate outflow cell index
        tgb_out = np.max(df_fl_mr.index)
        # pre-set outflow flow length value to 1 [m]
        df_fl.at[tgb_out, f_lower] = 1
        # iterate all cells in reversed order
        for tgb in reversed(df_fl_mr.index):
            # if cell is a routing or headwater cell
            if not ser_tgb_type_dummy.at[tgb]:
                # find real downstream cell
                jjnd_down = ser_tgb_down_nd.at[tgb]
                
                # SET LOWER CUMULATIVE FLOW LENGTH OF RECENT AS UPPER OF DOWNSTREAM CELL
                if tgb != tgb_out:
                    df_fl.at[tgb, f_lower] = df_fl.at[jjnd_down, f_upper]
                else:
                    df_fl.at[tgb, f_lower] = 0
                    
                # DECIDE ABOUT BEHAVIOR USING DOWNSTREAM PART OF RECENT AND
                # UPSTREAM PART OF DOWNSTREAM CELL
                # get downstream flow network flow length of RECENT cell
                if tgb != tgb_out: fl_fnw_down = df_fl_fnw.loc[tgb, f_corr_down]
                else:               fl_fnw_down = 0
                # if (1) downstream flow network flow length of RECENT cell is > 0
                # set downstream flow length to flow network flow length (key 1)
                # (no further distinction, as fl_fnw_down > 0 && fl_fnw_down_up <= 0
                # cannot exist due to the definition of df_fl_fnw)
                if fl_fnw_down > 0:
                    fl_down = df_fl_fnw.loc[tgb, f_corr_down]
                    df_fl_keys.at[tgb, f_down] = 1
                # if (2) downstream flow network flow length of RECENT cell is < 0
                # than a potential structural mismatch between model resolution flow
                # length and flow network flow length resulting from cell aggregation
                # has to be corrected. The downstream flow length is set to flow network
                # flow length (key -1)
                elif fl_fnw_down < 0:
                    fl_down = df_fl_fnw.loc[tgb, f_corr_down]
                    df_fl_keys.at[tgb, f_down] = -1
                # if (3) downstream flow network flow length of RECENT cell does not
                # exist (= 0), than model resolution flow length is used and further
                # distinction of cases is based on upstream flow network flow length
                # of DOWNSTREAM cell
                elif fl_fnw_down == 0:
                    # get upstream flow network flow length of DOWNSTREAM cell
                    # (except for outflow)
                    if tgb != tgb_out:
                        fl_fnw_down_up = df_fl_fnw.loc[jjnd_down, f_corr_up]
                    else:
                        fl_fnw_down_up = 0
                    # if (3.1) upstream flow network flow length of DOWNSTREAM cell
                    # does not exist (<= 0), than both cells have model resolution
                    # flow length and downstream flow length part is set to model 
                    # resolution flow length (key 100)
                    if fl_fnw_down_up <= 0:
                        fl_down = df_fl_mr.loc[tgb, f_corr_down]
                        df_fl_keys.at[tgb, f_down] = 100
                    # if (3.2) upstream flow network flow length of DOWNSTREAM
                    # cell exists (> 0) than there is a transition from downstream
                    # flow network to recent cell model resolution flow length
                    # and the difference of model resolution and flow network
                    # flow length is calculated (key -100).
                    else:
                        fl_down = df_fl_mr.loc[tgb, f_down] * 2 - fl_fnw_down_up
                        df_fl_keys.at[tgb, f_down] = -100
                        
                # CALCULATE UPSTREAM AND SUM OF FLOW LENGTH OF RECENT CELL
                # headwater cells: cell flow length = downstream part
                if ser_tgb_type_headw.at[tgb]:
                    df_fl.at[tgb, f_length] = fl_down
                # routing cells: cell flow length = downstream + upstream flow length
                else:
                    # get upstream flow network flow length of RECENT cell
                    fl_fnw_up = df_fl_fnw.loc[tgb, f_corr_up]
                    # if upstream flow network flow length of RECENT cell is > 0
                    # set upstream flow length to flow network flow length (key 1)
                    if fl_fnw_up > 0:
                        fl_up = fl_fnw_up
                        df_fl_keys.at[tgb, f_up] = 1
                    # if upstream flow network flow length is = 0 (< 0 cannot exist)
                    # set upstream flow length to model resolution flow length (key 100)
                    else:
                        fl_up = df_fl_mr.loc[tgb, f_corr_up]
                        df_fl_keys.at[tgb, f_up] = 100
                    # sum down- and upstream flow length parts (except for outflow)
                    if tgb != tgb_out: df_fl.at[tgb, f_length] = fl_down + fl_up
                    else:               df_fl.at[tgb, f_length] =           fl_up
                    
                # DO CORRECTIONS
                # if structural mismatches and transition values cannot be compensated
                # by upstream flow length part (flow length < 0), set flow length to
                # the threshold def_fl_strct_mism, a symbolic very small value
                if np.isin(df_fl_keys.at[tgb, f_down], [-1, -100]) \
                    and df_fl.at[tgb, f_length] <= def_fl_strct_mism:
                    df_fl.at[tgb, f_length] = def_fl_strct_mism
                # if flow length is unrealistic high (flow length > def_fl_upper_lim),
                # set flow length to the threshold def_fl_upper_lim
                if df_fl.at[tgb, f_length] > def_fl_upper_lim:
                    df_fl.at[tgb, f_length] = def_fl_upper_lim
                    
                # CALCULATE UPSTREAM CUMULATIVE FLOW LENGTH OF RECENT CELL
                # headwater cells: use lower cumulative flow length as upper
                # (not used in LARSIM, as there is no routing in head water cells)
                if ser_tgb_type_headw.at[tgb]:
                    df_fl.at[tgb, f_upper] = df_fl.at[tgb, f_lower]
                # routing cell, which is not outlet: calculate sum of downstream
                # cumulative flow length and flow length of recent cell
                elif tgb != tgb_out:
                    df_fl.at[tgb, f_upper] = df_fl.at[tgb, f_length] \
                                          + df_fl.at[tgb, f_lower]
                # routing cell, which is outlet: use flow length of recent cell
                else:
                    df_fl.at[tgb, f_upper] = df_fl.at[tgb, f_length]
            # if cell is a dummy cell
            else:
                # take value from downstream cell for upper and lower value
                df_fl.at[tgb, [f_lower, f_upper]] = df_fl.loc[ser_j_down.at[tgb], f_upper]
        return df_fl
    
    # %% Calculate cumulative flow length values respecting LARSIM conventions
    def calc_cum_ch_fl(df_fl, ser_tgb_up, ser_tgb_type_headw, ser_tgb_type_dummy):
        """
        This function calculates the cumulative flow length values respecting LARSIM
        conventions. 
        In elements with a difference of 1 [m] between upper and lower cumulative
        flow length (KMO and KMU) will the routing be ignored. Therefore, dummy 
        and head water elements shall be set to a difference of 1 between KMO and KMU 
        (KMO - KMU = 1). The function returns a pandas.DataFrame for KMO and KMU.
    
        JM 2021
    
        Arguments:
        -----------
        df_fl: pandas.DataFrame
            DataFrame of corresponding model resolution flow length values. The 
            DataFrame includes the model element ID as index and the following
            columns:
                - accumulative flow length at lower boundary of cell ('lower')
                - flow length value of cell ('length')
                - accumulative flow length at upper boundary of cell ('upper')
        ser_tgb_up: pandas.Series
            Series of corresponding upstream model element indices.
            Dummy elements are represented as empty array (e.g., []).
        ser_tgb_type_headw: pandas.Series
            Boolean Series, which identifies the headwater cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[1, 1, 0, 0], index=[1, 2, 3, 4], name='headwater',
                             dtype='bool'))
        ser_tgb_type_dummy: pandas.Series
            Boolean Series, which identifies the dummy cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[0, 0, 0, 0], index=[1, 2, 3, 4], name='dummy',
                             dtype='bool'))
    
        Returns:
        -----------
        df_cum_ch_fl: pandas.DataFrame
            DataFrame of corresponding runoff concentration parameters. The DataFrame
            includes the model element ID as index and the following columns:
                - corresponding lower cumulative flow length values (KMU) [m]
                - corresponding upper cumulative flow length values (KMO) [m]
        """
        # define internal string variables
        f_kmu       = 'kmu'
        f_kmo       = 'kmo'
        f_lower     = 'lower'
        f_upper     = 'upper'
        # Calculate Dummy adds for KMO and KMU
        # pre-allocate arrays for adds
        dummy_adds = pd.DataFrame(np.zeros((ser_tgb_up.shape[0], 2)),
                                  index=ser_tgb_up.index, columns=[f_lower, f_upper])
        # add of outlet is 1
        tgb_out = np.max(ser_tgb_up.index)
        dummy_adds.at[tgb_out, f_lower] = 1
        # iterate all cells
        for tgb in reversed(ser_tgb_up.index):
            # get upstream cell IDs
            tgb_up = ser_tgb_up.at[tgb]
            # lower add of upstream cell is upper add of recent cell
            dummy_adds.at[tgb_up, f_lower] = dummy_adds.at[tgb, f_upper]
            # get indices of upstream dummy cells
            tgb_up_dummys = ser_tgb_type_dummy.loc[tgb_up].index
            # if upstream cell is not a dummy cell, upper add = lower add
            dummy_adds.at[tgb_up, f_upper] = dummy_adds.loc[tgb_up, f_lower].values
            # if upstream cell is a dummy cell, upper add = upper add + 1
            dummy_adds.at[tgb_up_dummys, f_upper] \
                = dummy_adds.loc[tgb_up_dummys, f_upper].values + 1
        # Calculate head water adds
        headw_adds = pd.Series(np.zeros((ser_tgb_up.shape[0])), index=ser_tgb_up.index,
                               name=f_upper)
        headw_adds.at[ser_tgb_type_headw] = 1
        # Add Dummy and Head Water Adds
        ser_kmu = np.round(df_fl.loc[:, f_lower], 0) + dummy_adds.loc[:, f_lower]
        ser_kmo = np.round(df_fl.loc[:, f_upper], 0) + dummy_adds.loc[:, f_upper] \
                                                    + headw_adds
        # summarize parameters
        df_cum_ch_fl = pd.concat([ser_kmu, ser_kmo], axis=1)
        df_cum_ch_fl.columns = [f_kmu, f_kmo]
        
        return df_cum_ch_fl
    
    # %% calculate channel elevation differences
    def calc_ch_zdif(ser_zlower, df_fl,
                     ser_tgb_up_nd, ser_tgb_type_headw, ser_tgb_type_dummy,
                     def_sl_min=0.0001):
        """
        This function calculates the channel elevation differences and corrects them
        applying the LARSIM conventions. This means, that (1) a minimum channel slope
        is maintained. The slope value might be very small, but is not allowed to be
        zero. As there are LARSIM-internal rounding mechanisms, slope values smaller
        0.0001 mL/mZ have to be avoided. Additionally, (2) multiple upstream 
        neighbour elements have to be balanced, as only one elevation value can be
        applied to a single element. Potential conservation is achieved moving the
        elevation difference to the upstream element neighbours.
    
        JM 2021
    
        Arguments:
        -----------
        ser_zlower: pandas.Series
            Series of model elements' minimum elevation corresponding to the serie's
            ascending index. 
            (e.g., pd.Series([308.4, 341.0, 204.5, 133.8], index=[1, 2, 3, 4],
                             name='ser_zlower'))
        df_fl: pandas.DataFrame
            DataFrame of corresponding model resolution flow length values. The 
            DataFrame includes the model element ID as index and the following
            columns:
                - accumulative flow length at lower boundary of cell ('lower')
                - flow length value of cell ('length')
                - accumulative flow length at upper boundary of cell ('upper')
        ser_tgb_up_nd: pandas.Series
            Series of corresponding upstream model element indices ignoring dummy
            elements. These are represented as empty array (e.g., []).
        ser_tgb_type_headw: pandas.Series
            Boolean Series, which identifies the headwater cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[1, 1, 0, 0], index=[1, 2, 3, 4], name='headwater',
                             dtype='bool'))
        ser_tgb_type_dummy: pandas.Series
            Boolean Series, which identifies the dummy cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[0, 0, 0, 0], index=[1, 2, 3, 4], name='dummy',
                             dtype='bool'))
        def_sl_min: float (optional, default: 0.0001)
            minimum channel slope value to be maintained due to LARSIM-internal
            restrictions
            
        Returns:
        -----------
        df_ch_zdif: pandas.DataFrame
            DataFrame of corrected element elevation values. The DataFrame
            includes the model element ID as index and the following columns:
                - slope correction value [m] ('corr_sl')
                - balancing correction value [m] ('corr_bal')
                - corrected minimum channel elevation [m] ('lower_corr')
                - corrected channel elevation difference [m] ('ch_zdif')
                - corrected maximum channel elevation [m] ('upper_corr')
        """
        # define internal string variables
        f_length     = 'length'
        f_ch_zdif    = 'ch_zdif'
        f_corr_sl    = 'corr_sl'
        f_corr_bal   = 'corr_bal'
        f_lower_corr = 'lower_corr'
        f_upper_corr = 'upper_corr'
        # pre-allocate arrays
        df_ch_zdif = pd.DataFrame(
                np.zeros((ser_tgb_up_nd.shape[0], 5)) * np.nan, index=ser_tgb_up_nd.index,
                columns=[f_corr_sl, f_corr_bal, f_lower_corr, f_ch_zdif, f_upper_corr])
        # fill input columns (min and max elevation within cell)
        df_ch_zdif.lower_corr = ser_zlower
        # set dummy cell values to nan
        df_ch_zdif.at[ser_tgb_type_dummy, :] = np.nan
        # iterate all cells
        for tgb in reversed(ser_tgb_up_nd.index):
            # routing cells
            if not ser_tgb_type_dummy[tgb] and not ser_tgb_type_headw[tgb]:
                # get min elevation within cell
                zlower = df_ch_zdif.at[tgb, f_lower_corr]
                # find upstream cell ID number
                tgb_up_nd = ser_tgb_up_nd.at[tgb]
                # get elevation value for upstream cell
                zupper = df_ch_zdif.loc[tgb_up_nd, f_lower_corr]
                # calculate range threshold to prevent slope < def_sl_min
                zdif_sl_thr = def_sl_min * df_fl.at[tgb, f_length]
                # find cell pairs lower threshold slope
                sl_corr_bool = (zupper - zlower) <= zdif_sl_thr
                # if there is any, correct height differences lower than threshold
                if np.any(sl_corr_bool):
                    # get and set min elevation correction values
                    hd_corr = zdif_sl_thr - (zupper.loc[sl_corr_bool] - zlower)
                    df_ch_zdif.at[tgb_up_nd[sl_corr_bool], f_corr_sl] = hd_corr
                    # get and set max elevation correction values
                    zupper_sl_corr = zupper.loc[sl_corr_bool] + hd_corr
                    df_ch_zdif.at[tgb_up_nd[sl_corr_bool], f_lower_corr] = zupper_sl_corr
                    zupper.at[sl_corr_bool] = zupper_sl_corr.iloc[0]
                else:
                    df_ch_zdif.at[tgb_up_nd, f_corr_sl] = 0
                # if more than one upstream cells exist...
                if np.any(tgb_up_nd):
                    # ...calculate minimum value
                    zupper_min = np.nanmin(zupper)
                    df_ch_zdif.at[tgb, f_upper_corr] = zupper_min
                    df_ch_zdif.at[tgb_up_nd, f_lower_corr] = zupper_min
                    df_ch_zdif.at[tgb_up_nd, f_corr_bal] = zupper_min - zupper
                # if only one upstream cell exists take elevation value of it
                else:
                    df_ch_zdif.at[tgb_up_nd, f_corr_bal] = 0
                    df_ch_zdif.at[tgb, f_upper_corr] = zupper
        # calculate elevation range within cell
        df_ch_zdif.loc[:, f_ch_zdif] = \
                df_ch_zdif.loc[:, f_upper_corr] - df_ch_zdif.loc[:, f_lower_corr] 
        return df_ch_zdif
    
    # %% calculate runoff concentration parameters
    def calc_roconc_params(ser_ch_zmin, ser_zmax, ser_fl_ch_down, ser_fl_headw_len, 
                           ser_tgb_type_headw, ser_tgb_type_dummy,
                           cellsz, def_zmin_rout_fac=0.5, def_zmax_fac=1):
        """
        This function calculates the runoff concentration parameters needed for
        the retention time estimation using the Kirpich formula (Kirpich, 1940).
    
        JM 2021
    
        Arguments:
        -----------
        ser_ch_zmin: pandas.Series [m]
            Series of model elements' minimum channel elevation corresponding to
            the serie's ascending index. 
            (e.g., pd.Series([302.4, 330.0, 180.5, 120.8], index=[1, 2, 3, 4],
                             name='ser_ch_zmin'))
        ser_zmax: pandas.Series
            Series of model elements' maximum elevation corresponding to the 
            serie's ascending index. [m]
            (e.g., pd.Series([308.4, 341.0, 204.5, 133.8], index=[1, 2, 3, 4],
                             name='ser_zmax'))
        ser_fl_ch_down: pandas.Series [m]
            Series of model elements' downstream channel flow length parts 
            corresponding to the serie's ascending index. 
            (e.g., pd.Series([202.4, 120.0,  29.5,  13.8], index=[1, 2, 3, 4],
                             name='ser_fl_ch_down'))
        ser_fl_headw_len: pandas.Series
            Series of model elements' headwater flow length parts corresponding 
            to the serie's ascending index. [m]
            (e.g., pd.Series([110.4, 231.0, 204.5, 133.8], index=[1, 2, 3, 4],
                             name='ser_fl_headw_len'))
        ser_tgb_type_headw: pandas.Series
            Boolean Series, which identifies the headwater cells corresponding 
            to the serie's ascending index with True.
            (e.g., pd.Series(data=[1, 1, 0, 0], index=[1, 2, 3, 4], name='headw',
                             dtype='bool'))
        ser_tgb_type_dummy: pandas.Series
            Boolean Series, which identifies the dummy cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[0, 0, 0, 0], index=[1, 2, 3, 4], name='dummy',
                             dtype='bool'))
        cellsz: int
            Integer, which defines the model element edge length in [m] (e.g., 100)
        def_zmin_rout_fac: float (optional, default: 0.5)
            Factor to vary the lower elevation of runoff concentration between
            the minimum (0) and maximum (1) channel elevation of the element. By 
            default, the factor is set to the average elevation (0.5) [-] 
        def_zmax_fac: float (optional, default: 1)
            Factor to vary the upper elevation of runoff concentration between
            the minimum (0) and maximum (1) elevation of the element. By default,
            the factor is set to the maximum elevation (1) [-] 
    
        Returns:
        -----------
        df_roconc_params: pandas.DataFrame
            DataFrame of runoff concentration parameters. The DataFrame
            includes the model element ID as index and the following columns:
                - lower runoff concentration elevation [m] ('hut')
                - upper runoff concentration elevation [m] ('hot')
                - maximum runoff concentration flow length [km] ('tal')
        """
        # define internal string variables
        f_tal = 'tal'
        f_hut = 'hut'
        f_hot = 'hot'

        # calculate lower runoff concentration elevation
        # define HUT for head waters as low point of cell
        ser_hut = ser_ch_zmin + (ser_zmax - ser_ch_zmin) * def_zmin_rout_fac
        ser_hut.at[ser_tgb_type_headw] = ser_ch_zmin.loc[ser_tgb_type_headw]
        
        # calculate upper runoff concentration elevation
        ser_hot = ser_hut + (ser_zmax - ser_hut) * def_zmax_fac
        # correct negative and zero HOT-HUT
        zdif_corr_ii = np.round(ser_hot, 1) - np.round(ser_hut, 1) <= 0
        ser_hot.at[zdif_corr_ii] = ser_hut.loc[zdif_corr_ii] + 0.1
        
        # calculate maximum flow length
        # define TAL for cells with stream as mean of streight and diagonal line
        ser_tal = pd.Series(np.zeros(ser_hot.shape) + (np.sqrt(2) + 1) * cellsz / 4,
                            index=ser_hot.index, name=f_tal)
        # define TAL for head waters balancing flow length upstream values
        ser_tal.at[ser_tgb_type_headw] = \
            ser_fl_ch_down.loc[ser_tgb_type_headw] \
            + ser_fl_headw_len.loc[ser_tgb_type_headw]
        # convert from [m] to [km]
        ser_tal = ser_tal / 1000
        
        # summarize series
        df_roconc_params = pd.concat([ser_hut, ser_hot, ser_tal], axis=1)
        df_roconc_params.columns = [f_hut, f_hot, f_tal]
        df_roconc_params.at[ser_tgb_type_dummy, :] = np.nan
        
        return df_roconc_params
        
    # %% calculation
    # define key-words to identify element types
    str_headw   = 'headwater'
    str_routing = 'routing'
    str_dummy   = 'dummy'
    # define internal variables
    f_tgb      = 'tgb'
    f_tgb_down = 'tgb_down'
    f_tgb_type = 'tgb_type'
    f_tgb_a    = 'tgb_a'
    f_x        = 'x'
    f_y        = 'y'
    f_nrflv    = 'nrflv'
    f_ft       = 'ft'
    # define arcpy default field names
    f_pt_x     = 'POINT_X'
    f_pt_y     = 'POINT_Y'
    # calculate model network parameters
    if print_out: print('...import and pre-process data...')
    # Import model cell feature class attribute table and convert to pandas.DataFrame
    structarr_tgb_in = arcpy.da.FeatureClassToNumPyArray(
            name_tgb_par_p,
            [f_tgb, f_tgb_type, f_tgb_down, f_tgb_a, f_pt_x, f_pt_y, field_fl_mr,
             field_dem_max_mr, field_dem_min_mr, field_fl_fnw_mean_mr])
    df_tgb_in = pd.DataFrame(np.sort(structarr_tgb_in, order=f_tgb),
                             index=structarr_tgb_in[f_tgb])
    df_tgb_in = df_tgb_in.rename(columns={f_pt_x: f_x, f_pt_y: f_y})
    # convert string identifiers of model cells to logical arrays
    tgb_type_lookup, tgb_type_tgb_id = np.unique(df_tgb_in.loc[:, f_tgb_type],
                                                 return_inverse=True)
    ser_tgb_type_headw = pd.Series(
            tgb_type_tgb_id == np.nonzero(tgb_type_lookup == str_headw)[0][0],
            dtype=bool, index=df_tgb_in.index, name=str_headw)
    ser_tgb_type_routing = pd.Series(tgb_type_tgb_id == np.nonzero(
            tgb_type_lookup == str_routing)[0][0],
            dtype=bool, index=df_tgb_in.index, name=str_routing)
    ser_tgb_type_dummy = pd.Series(tgb_type_tgb_id == np.nonzero(
            tgb_type_lookup == str_dummy)[0][0],
            dtype=bool, index=df_tgb_in.index, name=str_dummy)
    # calculate upstream model element indices
    ser_tgb_up      = tc.get_upstream_idx(df_tgb_in.loc[:, f_tgb_down])
    # get up- and downstream model cell indices while ignoring dummy elements
    ser_tgb_down_nd = tc.get_downstream_idx_ign_dumm(
            df_tgb_in.loc[:, f_tgb_down], ser_tgb_type_dummy)
    ser_tgb_up_nd   = tc.get_upstream_idx_ign_dumm(
            df_tgb_in.loc[:, f_tgb_down], ser_tgb_type_headw, ser_tgb_type_dummy)
    
    # calculate model network parameters
    if print_out: print('...calculate model network parameters...')
    # redistribute model resolution flow length values at confluence points
    ser_fl = copy.deepcopy(df_tgb_in.loc[:, field_fl_mr])
    df_fl_mr = redistr_flowl_at_conflp(ser_fl, ser_tgb_down_nd, ser_tgb_up_nd, 
                                       ser_tgb_type_headw, ser_tgb_type_dummy)
    # redistribute flow network flow length values at confluence points
    # (including redistribution of very small flow length values)
    ser_fl_fnw = copy.deepcopy(df_tgb_in.loc[:, field_fl_fnw_mean_mr])
    df_fl_fnw = redistr_flowl_polyl_at_conflp(
            ser_fl_fnw, ser_tgb_down_nd, ser_tgb_up_nd,
            ser_tgb_type_headw, ser_tgb_type_dummy, cellsz)
    # merge flow length resulting from model resolution raster and flow network polylines
    df_fl = merge_fnw_and_mr_fl(df_fl_mr, df_fl_fnw,
                                df_tgb_in.loc[:, f_tgb_down], ser_tgb_down_nd, ser_tgb_type_headw,
                                ser_tgb_type_dummy, def_fl_upper_lim=def_fl_upper_lim,
                                def_fl_strct_mism=def_fl_strct_mism)
    # calculate cumulative flow length values respecting LARSIM conventions
    df_cum_ch_fl = calc_cum_ch_fl(df_fl, ser_tgb_up, ser_tgb_type_headw, ser_tgb_type_dummy)
    # calculate channel elevation differences
    df_ch_zdif = calc_ch_zdif(df_tgb_in.loc[:, field_dem_min_mr], df_fl, 
                              ser_tgb_up_nd, ser_tgb_type_headw, ser_tgb_type_dummy, 
                              def_sl_min=def_sl_min)
    # calculate slope for routing
    ser_ch_gef = tc.calc_ch_sl(df_ch_zdif.loc[:, 'ch_zdif'], df_fl.loc[:, 'length'],
                               ser_tgb_type_routing,
                               def_sl_excl_quant=def_sl_excl_quant)
    
    # calculate runoff concentration parameters
    if print_out: print('...calculate runoff concentration parameters...')
    df_roconc_params = calc_roconc_params(df_ch_zdif.lower_corr,
                                          df_tgb_in.loc[:, field_dem_max_mr],
                                          df_fl_mr.corr_down, df_fl.length,
                                          ser_tgb_type_headw, ser_tgb_type_dummy,
                                          cellsz, def_zmin_rout_fac=def_zmin_rout_fac, 
                                          def_zmax_fac=def_zmax_fac)
    
    # calculate routing parameters
    if print_out: print('...calculate routing parameters...')
    # calculate channel-forming discharge
    ser_ch_form_q = tc.calc_ch_form_q(df_tgb_in.loc[:, f_tgb_a], df_tgb_in.loc[:, f_tgb_down],
                                   q_spec=q_spec_ch, ser_q_in_corr=ser_q_in_corr)
    # calculate tripel trapezoid river cross section
    df_ttp = tc.calc_ttp(ser_ch_form_q, ser_tgb_type_routing, ch_est_method=ch_est_method,
                          def_bx=def_bx, def_bbx_fac=def_bbx_fac, def_bnm=def_bnm,
                          def_bnx=def_bnx, def_bnvrx=def_bnvrx, 
                          def_skm=def_skm, def_skx=def_skx)
        
    # calculate informative parameters
    if print_out: print('...calculate informative parameters...')
    # calculate inflow catchment size informative value
    ser_area_outfl = df_tgb_in.loc[:, f_tgb_a] + (cellsz**2) / (10**6)
    ser_area_outfl.at[~ser_tgb_type_routing] = 0
    # create names of elements
    ser_nrflv = pd.Series(df_tgb_in.shape[0]*'', index=df_tgb_in.index, name=f_nrflv)
    for tgb, el_type in df_tgb_in.loc[:, f_tgb_type].iteritems():
        ser_nrflv.at[tgb] = '{0:s}{1:05d}'.format(el_type[0].upper(), tgb)
    # calculate cell area value (FT)
    ser_ft = pd.Series(np.zeros(ser_tgb_type_dummy.shape),
                       index=ser_tgb_type_dummy.index, name=f_ft)
    ser_ft[~ser_tgb_type_dummy] = (cellsz**2) / (10**6)
    
    # summarize information to data frame
    if print_out: print('...summarize information...')
    # summarize Series to DataFrame
    fields = ['TGB','NRFLV','FT','HUT','HOT','TAL','X','Y','KMU','KMO','GEF',
        'HM','BM','BL','BR','BBL','BBR','BNM','BNL','BNR','BNVRL','BNVRR',
        'SKM','SKL','SKR','Kommentar_EZG-A','Kommentar_GBA']
    df_data_tgbdat = pd.concat([df_tgb_in.loc[:, f_tgb], ser_nrflv, ser_ft, df_roconc_params, 
                                df_tgb_in.loc[:, f_x].astype(np.int),
                                df_tgb_in.loc[:, f_y].astype(np.int),
                                df_cum_ch_fl, ser_ch_gef, df_ttp, ser_area_outfl,
                                ser_ch_form_q], axis=1)
    df_data_tgbdat.columns=fields
    # correct data for headwater and dummy catchments
    df_data_tgbdat.at[ser_tgb_type_headw,
                      ['GEF', 'HM','BM','BL','BR','BBL','BBR','BNM',
                       'BNL','BNR','BNVRL','BNVRR','SKM','SKL','SKR']] = np.nan
    df_data_tgbdat.at[ser_tgb_type_dummy,
                      ['HUT','HOT','TAL','GEF','HM','BM','BL','BR','BBL','BBR',
                       'BNM','BNL','BNR','BNVRL','BNVRR','SKM','SKL','SKR']] = np.nan
    
    return df_data_tgbdat, ser_tgb_down_nd, ser_ft, ser_area_outfl, ser_ch_form_q
    
# %% create cross section lines
def create_csl(path_fnw, path_ws_s, path_fa_hr, 
               def_cs_dist, def_cs_wmax_eval,
               path_gdb_out, name_csl='csl', name_fnw_o='fnw_o',
               def_river_a_in=0, def_cs_search_rad=5, print_out=False):
    """
    This function creates cross section lines with user defined spacing
    inbetween the lines and length of the cross sections perpendicular to the
    flow path. Additionally, the user may define a minimum catchment size from
    which cross sections are calculated. The function saves a polyline feature
    class containing the cross section lines and another one containing a 
    copy of the input flow network, which is used as reference for other
    functions.

    JM 2021

    Arguments:
    -----------
    path_fnw: str (e.g., 'c:\fnw.shp')
        path of the flow network feature class or shape file
    path_ws_s: str (e.g., 'c:\model_creation\ws_s')
        path of the model watershed domain polygon feature class
    path_fa_hr: str (e.g., 'c:\model_creation\fa_hr')
        path of the output extracted high resolution flow accumulation raster
    def_cs_dist: int (e.g., 200) [m]
        distance between transects along the flow path
    def_cs_wmax_eval: int (e.g., 600) [m]
        length of transects (incl. both sides of river)
    path_gdb_out: str (e.g., 'c:\model_creation.gdb')
        path of the output file geodatabase
    name_csl: str (optional, default: 'csl')
        name of the output cross section line feature class
    name_fnw_o: str (optional, default: 'fnw_o')
        name of reference flow network polyline feature class (copy of fnw)
    def_river_a_in: float (optional, default: 0) [km²]
        minimum inflowing catchment area defining a flow path where cross
        sections are created
    def_cs_search_rad: int (optional, default: 5) [m]
        search radius for transect identification
    print_out: boolean (optional, default: False)
        true if workprogress shall be print to command line

    Returns:
    -----------
    Saves the cross section line feature class.
    """
    
    # definitions
    # set intermediate feature class and raster names
    name_fnw                  = os.path.split(path_fnw)[1]
    name_fa_fnw_seg           = 'fa_fnw_seg'
    name_fnw_c                = 'fnw_c'
    name_fnw_dissolve         = 'fnw_dissolve'
    name_fnw_inters           = 'fnw_inters'
    name_fnw_gen_p            = 'fnw_gen_p'
    name_fnw_seg              = 'fnw_seg'
    name_fnw_seg_id           = 'fnw_seg_id'
    name_fnw_seg_id_jt        = 'fnw_seg_id_jt'
    name_fnw_seg_gen_p        = 'fnw_seg_gen_p'
    name_fnw_seg_buf          = 'fnw_seg_buf'
    name_fnw_seg_pfa          = 'fnw_seg_pfa'
    name_fnw_seg_gen_p_maxfat = 'fnw_seg_gen_p_maxfat'
    name_fnw_unspl            = 'fnw_unspl'
    # internal field names
    f_fnw_fid     = name_fnw     + '_fid'
    f_fnw_seg_fid = name_fnw_seg + '_fid'
    f_fnw_o_fid   = name_fnw_o   + '_fid'
    f_fnw_o_rp_id = name_fnw_o   + '_rp_fid'
    # arcpy field names
    f_shp_l       = 'Shape_Length'
    f_oid         = 'OBJECTID'
    f_join_fid    = 'JOIN_FID'
    f_target_fid  = 'TARGET_FID'
    f_orig_fid    = 'ORIG_FID'
    f_val         = 'Value'
    # arcpy method names
    method_first  = 'FIRST'
    method_max    = 'MAX'
    
    # calculations
    # set workspace
    arcpy.env.workspace = path_gdb_out
    # allow overwriting the outputs
    arcpy.env.overwriteOutput = True
    
    # Pre-process flow network
    if print_out: print('...pre-process flow network...')
    # clip flow network at model watershed domain
    arcpy.analysis.Clip(path_fnw, path_ws_s, name_fnw_c, None)
    # dissolve features
    arcpy.management.Dissolve(
            name_fnw_c, name_fnw_dissolve, '', None, 'SINGLE_PART', 'DISSOLVE_LINES')
    # merge features between confluence points respecting attributes
    arcpy.management.FeatureToLine(
            name_fnw_dissolve, name_fnw_inters, '', 'ATTRIBUTES')
    
    # Create flow network line segments
    if print_out: print('...create flow network line segments...')
    # generate points along flow network
    arcpy.GeneratePointsAlongLines_management(
            name_fnw_inters, name_fnw_gen_p, 'DISTANCE', int(def_cs_dist / 2), '', '')
    # split flow network lines at generated points
    #    def_pt_search_rad = 1 # [m]
    arcpy.SplitLineAtPoint_management(
            name_fnw_inters, name_fnw_gen_p, name_fnw_seg, None)
            # '{0:d} Meters'.format(def_pt_search_rad)
    
    # Get original Object-IDs from FGN1, join them and calculate buffer
    if print_out: print('...calculate buffer...')
    arcpy.SpatialJoin_analysis(
            name_fnw_seg, name_fnw_inters, name_fnw_seg_id,
            'JOIN_ONE_TO_MANY', 'KEEP_ALL',
            '{0} {0} false true true 8 Double 0 0, First, #, {1}, {0}, -1, -1;'.format(
            f_shp_l, name_fnw_seg), 'WITHIN', '', '')
    stat_expr = '{0} {1}'.format(f_join_fid, method_first)
    arcpy.analysis.Statistics(
            name_fnw_seg_id, name_fnw_seg_id_jt, stat_expr, f_target_fid)
    f_first_jfid = '{0}_{1}'.format(method_first, f_join_fid)
    arcpy.management.AlterField(
            name_fnw_seg_id_jt, f_first_jfid, f_fnw_fid, '', '', '',
            'NULLABLE', 'CLEAR_ALIAS')
    arcpy.management.AlterField(
            name_fnw_seg_id_jt, f_target_fid, f_fnw_seg_fid, '', '', '',
            'NULLABLE', 'CLEAR_ALIAS')
    arcpy.management.JoinField(
            name_fnw_seg, f_oid, name_fnw_seg_id_jt, f_fnw_seg_fid, f_fnw_fid)
    arcpy.analysis.Buffer(
            name_fnw_seg, name_fnw_seg_buf, str(def_cs_search_rad),
            'FULL', 'FLAT', 'NONE', '', 'PLANAR')
    
    # Calculate zonal statistics for maximum flow accumulation per eval. point,
    # create evaluation points, make table out of it and delete not necessary fields
    if print_out: print('...calculate zonal statistics...')
    fa_fnw_seg = arcpy.ia.ZonalStatistics(
            name_fnw_seg_buf, f_orig_fid, path_fa_hr, 'MAXIMUM', 'DATA')
    fa_fnw_seg.save(name_fa_fnw_seg)
    arcpy.management.GeneratePointsAlongLines(
            name_fnw_seg, name_fnw_seg_gen_p, 'DISTANCE',
            str(def_cs_search_rad + 2), '', '')
    arcpy.ga.ExtractValuesToTable(
            name_fnw_seg_gen_p, name_fa_fnw_seg, name_fnw_seg_pfa, '', 'ADD_WARNING_FIELD')
    arcpy.management.DeleteField(name_fnw_seg_pfa, 'SrcID_Rast')
    
    # Join original evaluation point object ID, rename field and calculate maximum
    # flow accumulation per segment
    if print_out: print('...calculate maximum flow accumulation per segment...')
    arcpy.JoinField_management(
            name_fnw_seg_gen_p, f_oid, name_fnw_seg_pfa, 'SrcID_Feat', f_val)
    arcpy.AlterField_management(
            name_fnw_seg_gen_p, f_orig_fid, f_fnw_seg_fid, '', 'LONG', '4',
            'NULLABLE', 'CLEAR_ALIAS')
    stat_expr = '{0} {1}'.format(f_val, method_max)
    arcpy.Statistics_analysis(
            name_fnw_seg_gen_p, name_fnw_seg_gen_p_maxfat,
            stat_expr, f_fnw_seg_fid)
    
    # Join original segment object ID, select elements smaller flow accumulation threshold,
    # delete them and merge lines to segments between confluence points
    if print_out: print('...create line segments between confluence points...')
    f_max_val = '{0}_{1}'.format(method_max, f_val)
    arcpy.JoinField_management(
            name_fnw_seg, f_oid, name_fnw_seg_gen_p_maxfat, f_fnw_seg_fid, f_max_val)
    # get high resolution cell size
    cell_sz_x_obj = arcpy.GetRasterProperties_management(path_fa_hr, 'CELLSIZEX')
    cell_sz_x = np.int32(cell_sz_x_obj.getOutput(0))
    def_river_cellnb = np.int64(def_river_a_in * 10**6 / cell_sz_x)
    sel_expr = '"{0:s}" >= {1:d}'.format(f_max_val, def_river_cellnb)
    fnw_seg_sel = arcpy.SelectLayerByAttribute_management(
            name_fnw_seg, 'NEW_SELECTION', sel_expr)
    arcpy.UnsplitLine_management(fnw_seg_sel, name_fnw_unspl, f_fnw_fid, '')
    # if field f_fnw_orig_fid does not exist, create and calculate it
    if not arcpy.ListFields(name_fnw_inters, f_fnw_o_fid):
        arcpy.AddField_management(
                name_fnw_unspl, f_fnw_o_fid, 'LONG', '', '', '', '',
                'NULLABLE', 'NON_REQUIRED', '')
        arcpy.CalculateField_management(
                name_fnw_unspl, f_fnw_o_fid, '!{0}!'.format(f_oid), 'PYTHON3', '')
    
    # Generate transects along line features, which are longer than defined 
    # distance between transects
    if print_out: print('...create transects along line features...')
    sel_expr = '{0:s} > {1:d}'.format(f_shp_l, def_cs_dist)
    arcpy.SelectLayerByAttribute_management(
            name_fnw_unspl, 'NEW_SELECTION', sel_expr, '')
    arcpy.CopyFeatures_management(name_fnw_unspl, name_fnw_o, '', '', '', '')
    arcpy.GenerateTransectsAlongLines_management(
            name_fnw_o, name_csl, str(def_cs_dist), str(def_cs_wmax_eval), 'NO_END_POINTS')
    arcpy.AlterField_management(
            name_csl, f_orig_fid, f_fnw_o_rp_id, '', '', '4', 'NULLABLE', 'CLEAR_ALIAS')
    
# %% create profile.dat based on user-defined cross sections
def df_profdat_from_cs(
        path_fnw, path_fnw_o, path_csl, path_dem_hr, path_fa_hr, path_tgb_s, 
        ser_tgb_down, ser_tgb_down_nd, ser_tgb_type_routing,
        ser_ch_form_q, ser_pef_bm, ser_pef_hm, 
        path_gdb_out, name_profile_par='profile_par',
        name_ch_fit_coords='ch_fit_coords', name_bcwsl='bcwsl', 
        def_cs_wmax_eval=600, def_cs_intp_buf_dist=1, def_ch_w=0.5, def_ch_h=0.5,
        def_ch_wres=0.05, def_cs_hmax_eval=10, def_lam_hres=0.1,
        def_ch_vmin=0.5, def_ch_vmax=3.0, def_ch_wmax_eval=40, 
        def_chbank_slmin=0.1, def_ch_hmin=0.2, def_ch_hmin_eval=0.1, 
        def_profdat_decmax=2, 
        ctrl_show_plots=False, ctrl_save_plots=False, 
        ser_ft=None, ser_area_outfl=None, 
        def_ch_hmax_eval=None, path_plots_out=None,
        print_out=False):
    """
    This function extracts and summarizes all parameters for user-defined cross
    sections and returns a DataFrame of them. Additionally, a Series is returned, 
    that represents the cross section line IDs, which are allocated to every 
    routing model element.

    JM 2021

    Arguments:
    -----------
    path_fnw: str (e.g., 'c:\model_creation.gdb\fnw.shp')
        path of the flow network feature class or shape file
    path_fnw_o: str (e.g., 'c:\model_creation.gdb\fnw_o')
        path of reference flow network polyline feature class (copy of fnw)
    path_csl: str (e.g., 'c:\model_creation.gdb\csl')
        path of the output cross section line feature class
    path_dem_hr: str (e.g., 'c:\model_creation.gdb\dem_hr')
        path of the output watershed polygon feature class
    path_fa_hr: str (e.g., 'c:\model_creation.gdb\fa_hr')
        path of the output extracted high resolution flow accumulation raster
    path_tgb_s: str (e.g., 'c:\model_creation.gdb\tgb_s')
        path of the output model element polygon feature class
    ser_rout_tgb: pandas.Series
        series of routing type model element ID numbers
    ser_tgb_down: pandas.Series
        Series of downstream model element indices corresponding to the serie's
        ascending index. The last value is outlet, identified with a zero.
        The outlet will be neglected in calculations.
        (e.g., pd.Series([2, 4, 4, 0], index=[1, 2, 3, 4], name='tgb_down'))
    ser_tgb_down_nd: pandas.Series
        Series of corresponding downstream model element indices ignoring dummy
        elements. Model outlet remains -1 and dummy elements are represented as 0.
    ser_tgb_type_routing: pandas.Series
        Boolean Series, which identifies the routing cells corresponding to the
        serie's ascending index with True.
        (e.g., pd.Series(data=[0, 0, 1, 1], index=[1, 2, 3, 4],
                         name='routing', dtype='bool'))
    ser_ch_form_q: pandas.Series
        Series of elements' channel-forming discharge at the corresponding
        model element ID in the serie's index. 
    ser_pef_bm, ser_pef_hm: pandas.Series
        Series of estimated channel width ('bm') and depth ('wm')
    path_gdb_out: str (e.g., 'c:\model_creation.gdb')
        path of the output file geodatabase
    name_profile_par: str (optional, default: 'profile_par')
        name of parameter table for bankful discharge
    name_ch_fit_coords: str (optional, default: 'ch_fit_coords')
        name of coordinate table for bankful discharge line
    name_bcwsl: str (optional, default: 'bcwsl')
        name of bankful channel water surface line
    def_cs_wmax_eval: int (optional, default: 600) [m]
        Length of the automatically generated cross sections perpendicular
        to the flow accumulation flow network. It should cover the valley
        at least until the estimated maximum water depth.
    def_cs_hmax_eval: float (optional, default: 10) [m]
        maximum height of cross section evaluation
    def_ch_wmax_eval: float (optional, default: 40) [m]
        estimated maximum channel width
    def_ch_hmin_eval: float (optional, default: 0.1) [m]
        minimum height of channel evaluation
    def_ch_hmax_eval: float (optional, default: None)
        maximum height of channel evaluation (used to limit y-axis of plot)
    def_ch_hmin: float (optional, default: 0.2, must be >= 0.2) [m]
        minimum channel depth threshold for channel identification
    def_ch_vmin: float (optional, default: 0.5) [m/s]
        minimum reasonable flow velocity
    def_ch_vmax: float (optional, default: 3.0) [m/s]
        maximum reasonable flow velocity
    def_chbank_slmin: float (optional, default: 0.1) [dH/dL]
        minimum riverbank slope threshold for channel identification
    def_ch_wres: float (optional, default: 0.05)
        horizontal resolution of interpolated points within channel
    def_lam_hres: float (optional, default: 0.1) [m]
        spacing between evaluation lamellae
    def_profdat_decmax: int (optional, default: 2) [-]
        decimal numbers allowed in the file profile.dat
    def_cs_intp_buf_dist: int (e.g., 1) [m]
        cross section intersection points' buffer distance
    def_ch_w, def_ch_h: float (optional, default: 0.5, 0.5) [m]
        artificial channel width (w) and depth (h), added to continuiously
        descending cross sections
    ctrl_show_plots: boolean (optional, default: False) [-]
        (de-)activate pop-up of figures
    ctrl_save_plots: boolean (optional, default: False) [-]
        (de-)activate export of figures as files
    ser_ft: pandas.Series (optional, default: None) [km²]
        model element subcatchment area
    ser_area_outfl: pandas.Series (optional, default: None) [km²]
        sum of upstream model elements' area
    path_plots_out: str (optional, default: None)
        path where plots are stored (e.g., 'c:\model_creation\fig')
    print_out: boolean (optional, default: False)
        true if workprogress shall be print to command line

    Returns:
    -----------
    df_profdat_par: pandas.DataFrame
        DataFrame containing all parameters calculated for a cross section 
        profile ID. These are:
        - csl_fid: cross section feature ID
        - h, a, p, wsl: water level depth (h) and width (wsl) as well as cross
          section area (a) and wetted perimeter (p) of bankful discharge
        - ll_ii: ID of the lamella determining the bankful discharge
        - a_ll, p_ll, wslm_ll: mean water surface level width (wslm), cross
          section area (a_ll) and wetted perimeter (p_ll) per lamellae
        The index is the allocated model element ID (tgb).
    ser_tgb_csl: pandas.Series
        Series of cross section ID numbers, which are allocated to all 
        routing model elements in the model structure (index: tgb)
    """
    
    # %% get cross section GIS data information
    def get_cs_gisdata(path_fnw, path_fnw_o, path_csl, path_dem_hr, path_fa_hr, 
                       path_tgb_s, df_tgb_in, path_gdb_out, 
                       def_cs_wmax_eval=600, def_cs_intp_buf_dist=1,
                       print_out=False):
        """
        This function intersects cross section lines with the model element 
        polygons and the flow network (mid-points of cross sections and 
        non-intended intersections with other channels of the flow network). 
        Furthermore, it collects necessary elevation information for the cross 
        section lines itself and the created intersection points.
        The function returns pandas.DataFrames with the collected information.
    
        JM 2021
    
        Arguments:
        -----------
        path_fnw: str (e.g., 'c:\fnw.shp')
            path of the flow network feature class or shape file
        path_fnw_o: str (e.g., 'c:\model_creation\fnw_o')
            path of reference flow network polyline feature class (copy of fnw)
        path_csl: str (e.g., 'c:\model_creation\csl')
            path of the output cross section line feature class
        path_dem_hr: str (e.g., 'c:\model_creation\dem_hr')
            path of the output watershed polygon feature class
        path_fa_hr: str (e.g., 'c:\model_creation\fa_hr')
            path of the output extracted high resolution flow accumulation raster
        path_tgb_s: str (e.g., 'c:\model_creation\tgb_s')
            path of the output model element polygon feature class
        ser_rout_tgb: pandas.Series
            series of routing type model element ID numbers
        path_gdb_out: str (e.g., 'c:\model_creation.gdb')
            path of the output file geodatabase
        def_cs_wmax_eval: int (optional, default: 600) [m]
            length of transects (incl. both sides of river)
        def_cs_intp_buf_dist: int (optional, default: 1) [m]
            cross section intersection points' buffer distance
        print_out: boolean (optional, default: False)
            true if workprogress shall be print to command line
    
        Returns:
        -----------
        df_csp: pandas.DataFrame
            DataFrame containing all information for the cross section line:
            - csl_fid: cross section line feature ID
            - x, y, z: X-, Y-, and Z-coordinates for points along the cross section
        df_intp: pandas.DataFrame
            DataFrame containing all information for the cross sections' 
            intersection points with potential other flow channel around:
            - csl_fid: cross section line feature ID
            - x, y, z: X-, Y-, and Z-coordinates for mid-point
            - fa:      flow accumulation value at intersection point
        df_intp_tgb: pandas.DataFrame
            DataFrame containing all information for the cross sections' mid-point
            (is equal to the intersection point with the original flow channel):
            - csl_fid: cross section line feature ID
            - x, y, z: X-, Y-, and Z-coordinates for mid-point
            - tgb:     model element ID number
            - fa:      flow accumulation value at intersection point
        """
        # workaround for bug in arcpy.ExtractMultiValuesToPoints if only one feature exists
        def workar_extrmultvaltopt(path_dem_hr, path_gdb_out, name_cs_intp,
                                   f_rasval='RASTERVALU', f_z='z'):
            path_cs_intp = path_gdb_out + name_cs_intp
            path_cs_intp_z = path_cs_intp + '_z'
            arcpy.sa.ExtractValuesToPoints(name_cs_intp, path_dem_hr, path_cs_intp_z,
                                           'NONE', 'VALUE_ONLY')
            arcpy.management.AlterField(path_cs_intp_z, f_rasval, f_z, '', 'FLOAT', 4,
                                        'NULLABLE', 'CLEAR_ALIAS')
            arcpy.management.Delete(name_cs_intp)
            arcpy.CopyFeatures_management(path_cs_intp_z, name_cs_intp, '', '', '', '')
            arcpy.AddXY_management(name_cs_intp)
        
        # definition section
        # define paths of intermediates in working geodatabase
        # cs:   cross section
        # ip:   intersection point
        # fnw:  flow network
        # csl:  cross section line
        # midp: mid-point
        name_fnw                 = os.path.split(path_fnw)[1]
        name_fnw_o               = os.path.split(path_fnw_o)[1]
        name_csl                 = os.path.split(path_csl)[1]
        name_csl_bu              = name_csl + '_bu'      # backup copy of csl
        name_cs_intp_mult        = 'cs_intp_mult'        # multi-part ips of cs with fnw
        name_cs_intp             = 'cs_intp'             # single-part ips of cs with fnw
        name_cs_midp_near_tab    = 'cs_midp_near_tab'    # table with closest ip to midp
        name_cs_intp_buf         = 'cs_inp_buf'          # cs ips' buffer
        name_fa_max_zonal        = 'fa_max_zonal'        # max fa zonal statistics raster at cs ips
        name_cs_midp_minrank_tab = 'cs_midp_minrank_tab' # min distance rank per csl table
        name_cs_midp             = 'cs_midp'             # cs midps
        name_cs_intp_other       = 'cs_inp_other'        # cs ips with other stream section
        name_cs_intp_same        = 'cs_inp_same'         # cs ips with same stream section
        name_cs_intp_tgb         = 'cs_intp_tgb'         # cs ips with model cell ID number
        name_csp                 = 'csp'                 # cs vertices including elevation values
        name_csp_z_tab           = 'csp_z_tab'           # cs vertice table including z-values
        # internal field names
        f_fnw_fid            = name_fnw   + '_fid'
        f_fnw_o_fid          = name_fnw_o + '_fid'
        f_fnw_o_rp_id        = name_fnw_o + '_rp_fid'
        f_fnw_o_ip_fid       = name_fnw_o + '_ip_fid'
        f_csl_fid            = name_csl + '_fid'
        f_csl_p_fid          = 'csl_p_fid'
        f_cs_midp_fid        = name_cs_midp + '_fid'
        f_cs_midp_o_fid      = name_cs_midp + '_o_fid'
        f_cs_midp_csp_dl     = name_cs_midp + '_csp_dl'
        f_cs_midp_csp_dlr    = name_cs_midp + '_csp_dlr'
        f_cs_midp_csp_dlrmin = name_cs_midp + '_csp_dlrmin'
        f_csp_fid            = name_csp + '_fid'
        f_csp_c_fid          = 'csp_c_fid'
        f_fa                 = 'fa'
        f_x                  = 'x'
        f_y                  = 'y'
        f_z                  = 'z'
        # arcpy field names
        f_oid       = 'OBJECTID'
        f_orig_fid  = 'ORIG_FID'
        f_val       = 'Value'
        f_in_fid    = 'IN_FID'
        f_near_fid  = 'NEAR_FID'
        f_near_dist = 'NEAR_DIST'
        f_near_rank = 'NEAR_RANK'
        f_rasval    = 'RASTERVALU'
        f_pt_x      = 'POINT_X'
        f_pt_y      = 'POINT_Y'
        f_pt_z      = 'POINT_Z'
        f_src_fid   = 'SrcID_Feat'
        # arcpy method names
        method_min  = 'MIN'
        
        # calculation section
        # set workspace
        arcpy.env.workspace = path_gdb_out
        # Calculate cross section line intersection points
        if print_out: print('Calculate cross section line intersection points...')
        arcpy.CopyFeatures_management(path_csl, name_csl_bu, '', '', '', '')
        # Create midpoints of cross section lines and rename field
        if print_out: print('...create midpoints of cross section lines...')
        arcpy.FeatureVerticesToPoints_management(name_csl, name_cs_midp, 'MID')
        arcpy.AlterField_management(name_cs_midp, f_orig_fid, f_cs_midp_fid,
                                    '', '', '4', 'NULLABLE', 'CLEAR_ALIAS')
        # Create intersection points of flow network and cross sections and rename fields
        if print_out: print('...create intersection points of flow network and cross sections...')
        arcpy.Intersect_analysis(r'{0} #;{1} #'.format(path_fnw_o, name_csl),
                                 name_cs_intp_mult, 'ALL', '', 'POINT')
        arcpy.AlterField_management(name_cs_intp_mult, f_fnw_o_fid, f_fnw_o_ip_fid,
                                    '', '', '4', 'NULLABLE', 'CLEAR_ALIAS')
        arcpy.MultipartToSinglepart_management(name_cs_intp_mult, name_cs_intp)
        arcpy.AlterField_management(name_cs_intp, f_orig_fid, f_csp_c_fid,
                                    '', 'LONG', '4', 'NULLABLE', 'CLEAR_ALIAS')
        arcpy.AlterField_management(name_cs_intp, 'FID_{0}'.format(name_csl), f_csl_fid,
                                    '', 'LONG', '4', 'NULLABLE', 'CLEAR_ALIAS')
        # Find closest intersection point, rename fields and join table to midpoint feature class
        if print_out: print('...find closest intersection point...')
        arcpy.GenerateNearTable_analysis(
                name_cs_midp, name_cs_intp, name_cs_midp_near_tab, str(def_cs_wmax_eval / 2),
                'NO_LOCATION', 'NO_ANGLE', 'ALL', '0', 'PLANAR')
        arcpy.AlterField_management(
                name_cs_midp_near_tab, f_in_fid, f_cs_midp_o_fid,
                '', '', '4', 'NULLABLE', 'CLEAR_ALIAS')
        arcpy.AlterField_management(
                name_cs_midp_near_tab, f_near_fid, f_csp_fid,
                '', '', '4', 'NULLABLE', 'CLEAR_ALIAS')
        arcpy.AlterField_management(
                name_cs_midp_near_tab, f_near_dist, f_cs_midp_csp_dl,
                '', '', '8', 'NULLABLE', 'CLEAR_ALIAS')
        arcpy.AlterField_management(
                name_cs_midp_near_tab, f_near_rank, f_cs_midp_csp_dlr,
                '', '', '4', 'NULLABLE', 'CLEAR_ALIAS')
        arcpy.JoinField_management(
                name_cs_midp_near_tab, f_cs_midp_o_fid, name_cs_midp, f_oid, f_cs_midp_fid)
        # Join closest intersection points' table to intersection points, rename field,
        # and delete intersection points with other CSL
        if print_out: print('...delete intersection points with other CSL...')
        arcpy.JoinField_management(
                name_cs_midp_near_tab, f_csp_fid, name_cs_intp, f_oid, f_csl_fid)
        arcpy.AlterField_management(
                name_cs_midp_near_tab, f_csl_fid, f_csl_p_fid,
                '', '', '4', 'NULLABLE', 'CLEAR_ALIAS')
        sel_expr = '{0} <> {1}'.format(f_cs_midp_fid, f_csl_p_fid)
        cs_midp_near_tab_sel = arcpy.SelectLayerByAttribute_management(
                name_cs_midp_near_tab, 'NEW_SELECTION', sel_expr, '')
        arcpy.DeleteRows_management(cs_midp_near_tab_sel)
        # Calculate buffer and zonal statistics of flow accumulation and extract values
        # at intersection points
        if print_out: print('...calculate zonal statistics of flow accumulation...')
        arcpy.Buffer_analysis(
                name_cs_intp, name_cs_intp_buf, str(def_cs_intp_buf_dist),
                'FULL', 'ROUND', 'NONE', '', 'PLANAR')
        fa_max_zonal = arcpy.ia.ZonalStatistics(
                name_cs_intp_buf, f_oid, path_fa_hr, 'MAXIMUM', 'DATA')
        fa_max_zonal.save(name_fa_max_zonal)
        arcpy.gp.ExtractMultiValuesToPoints_sa(
                name_cs_intp,  r'{0} {1}'.format(name_fa_max_zonal, f_fa), 'NONE')
        # Join distance rank of intersection points to rank table and copy intersection
        # points with another flow network polyline to feature class
        if print_out: print('...define intersection points with another CSL...')
        arcpy.JoinField_management(
                name_cs_intp, f_oid, name_cs_midp_near_tab, f_csp_fid, f_cs_midp_csp_dlr)
        sel_expr = '{0} <> {1}'.format(f_fnw_o_ip_fid, f_fnw_o_rp_id)
        cs_intp_ir_sel = arcpy.SelectLayerByAttribute_management(
                name_cs_intp, 'NEW_SELECTION', sel_expr, '')
        arcpy.CopyFeatures_management(cs_intp_ir_sel, name_cs_intp_other, '', '', '', '')
        arcpy.DeleteRows_management(cs_intp_ir_sel)
        # calculate minimum distance rank per cross section line, alter field,
        # and join minimum rank to intersection points
        # copy intersection points with identical flow network polyline to feature class
        if print_out: print('...define intersection points with identical CSL...')
        stat_expr = '{0} {1}'.format(f_cs_midp_csp_dlr, method_min)
        arcpy.Statistics_analysis(
                name_cs_intp, name_cs_midp_minrank_tab, stat_expr, f_csl_fid)
        f_min_cs_midp_csp_dlr = '{0}_{1}'.format(method_min, f_cs_midp_csp_dlr)
        arcpy.AlterField_management(
                name_cs_midp_minrank_tab, f_min_cs_midp_csp_dlr, f_cs_midp_csp_dlrmin,
                '', '', '8', 'NULLABLE', 'CLEAR_ALIAS')
        arcpy.JoinField_management(
                name_cs_intp, f_csl_fid, name_cs_midp_minrank_tab, f_csl_fid, f_cs_midp_csp_dlrmin)
        sel_expr = '{0} <> {1}'.format(f_cs_midp_csp_dlr, f_cs_midp_csp_dlrmin)
        cs_intp_sel_rmin = arcpy.SelectLayerByAttribute_management(
                name_cs_intp, 'NEW_SELECTION', sel_expr, '')
        arcpy.CopyFeatures_management(cs_intp_sel_rmin, name_cs_intp_same, '', '', '', '')
        arcpy.DeleteRows_management(cs_intp_sel_rmin)
        # intersect TGB shapes with cross section points and calculate TGB-ID field
        if print_out: print('...add model cell ID to intersection points...')
        arcpy.Intersect_analysis(
                r'{0} #;{1} #'.format(name_cs_intp, path_tgb_s), name_cs_intp_tgb,
                'NO_FID', '', 'POINT')
        # Get X-,Y-, and Z-coordinates for crossection points
        if print_out: print('...get x-, y-, and z-coordinates for crossection points...')
        # If Z-Field Exists, Drop Field
        if arcpy.ListFields(name_cs_intp_other, f_z): 
            arcpy.DeleteField_management(name_cs_intp_other, f_z)
        if arcpy.ListFields(name_cs_intp_same, f_z): 
            arcpy.DeleteField_management(name_cs_intp_same, f_z)
        if arcpy.ListFields(name_cs_intp_tgb, f_z): 
            arcpy.DeleteField_management(name_cs_intp_tgb, f_z)
        # Extract Multi Values to Z-field in Intersection Points
        # workaround for bug in ExtractMultiValuesToPoints if only one feature exists
        # arcpy.sa.ExtractMultiValuesToPoints(name_cs_intp_other, path_dem_hr + ' Z', 'NONE')
        # arcpy.sa.ExtractMultiValuesToPoints(name_cs_intp_same, path_dem_hr + ' Z', 'NONE')
        # arcpy.sa.ExtractMultiValuesToPoints(name_cs_intp_tgb, path_dem_hr + ' Z', 'NONE')
        workar_extrmultvaltopt(path_dem_hr, path_gdb_out, name_cs_intp_other,
                               f_rasval=f_rasval, f_z=f_z)
        workar_extrmultvaltopt(path_dem_hr, path_gdb_out, name_cs_intp_same,
                               f_rasval=f_rasval, f_z=f_z)
        workar_extrmultvaltopt(path_dem_hr, path_gdb_out, name_cs_intp_tgb,
                               f_rasval=f_rasval, f_z=f_z)
        
        if print_out: print('Create Points along Cross-section...')
        # Generate Points Along Lines
        if print_out: print('...generate points along lines...')
        cell_sz_x_obj = arcpy.GetRasterProperties_management(path_dem_hr, 'CELLSIZEX')
        cell_sz_x = np.int32(cell_sz_x_obj.getOutput(0))
        arcpy.GeneratePointsAlongLines_management(
                path_csl, name_csp, 'DISTANCE', str(cell_sz_x), '', 'END_POINTS')
        arcpy.AlterField_management(
                name_csp, f_orig_fid, f_csl_fid, '', '', '', 'NULLABLE', 'CLEAR_ALIAS')
        # Extract Elevation Values to Z-field in Cross Section Points
        if print_out: print('...add Z-coordinate...')
        arcpy.ExtractValuesToTable_ga(
                name_csp, path_dem_hr, name_csp_z_tab, '', 'ADD_WARNING_FIELD')
        arcpy.JoinField_management(
                name_csp, f_oid, name_csp_z_tab, f_src_fid, f_val)
        arcpy.AlterField_management(
                name_csp, f_val, f_z, '', '', '', 'NULLABLE', 'CLEAR_ALIAS')
        # Add XY Coordinates
        if print_out: print('...add X- and Y-coordinates...')
        arcpy.AddXY_management(name_csp)
        arcpy.DeleteField_management(name_csp, f_fnw_o_rp_id)
        drop_f_expr = ';'.join([f_cs_midp_csp_dlrmin, f_fnw_o_ip_fid,
                                f_fnw_o_rp_id, f_fnw_fid, f_csp_c_fid, f_pt_z])
        arcpy.DeleteField_management(name_cs_intp_other, drop_f_expr)
        drop_f_expr += ';{0}'.format(f_cs_midp_csp_dlr)
        arcpy.DeleteField_management(name_cs_intp_same, drop_f_expr)
        arcpy.DeleteField_management(name_cs_intp_tgb, drop_f_expr)
    
        # import cross section information
        if print_out: print('...import cross section data...')
        # import cross section tables
        csp_fields     = [f_csl_fid, f_oid, f_pt_x, f_pt_y, f_z]
        arr_csp        = arcpy.da.FeatureClassToNumPyArray(name_csp, csp_fields)
        cs_intp_fields = csp_fields + [f_fa]
        arr_intp_other = arcpy.da.FeatureClassToNumPyArray(name_cs_intp_other, cs_intp_fields)
        arr_intp_same  = arcpy.da.FeatureClassToNumPyArray(name_cs_intp_same,  cs_intp_fields)
        cs_intp_tgb_fields = cs_intp_fields + [f_tgb]
        arr_intp_tgb   = arcpy.da.FeatureClassToNumPyArray(name_cs_intp_tgb,   cs_intp_tgb_fields)
        # convert structured arrays to pandas DataFrames
        csp_fields.remove(f_oid)
        df_csp         = pd.DataFrame(arr_csp[csp_fields],
                                      index=arr_csp[f_oid]).rename(
                                              columns={f_pt_x: f_x, f_pt_y: f_y})
        cs_intp_tgb_fields.remove(f_oid)
        df_intp_tgb    = pd.DataFrame(arr_intp_tgb[cs_intp_tgb_fields],
                                      index=arr_intp_tgb[f_oid]).rename(
                                              columns={f_pt_x: f_x, f_pt_y: f_y})
        cs_intp_fields.remove(f_oid)
        df_intp_other  = pd.DataFrame(arr_intp_other[cs_intp_fields],
                                      index=arr_intp_other[f_oid]).rename(
                                              columns={f_pt_x: f_x, f_pt_y: f_y})
        df_intp_same   = pd.DataFrame(arr_intp_same[cs_intp_fields],
                                      index=arr_intp_same[f_oid]).rename(
                                              columns={f_pt_x: f_x, f_pt_y: f_y})
        df_intp        = pd.concat([df_intp_other, df_intp_same], axis=0)
        # sort lists by cross section line ID
        df_csp      = df_csp.sort_values(f_csl_fid, axis=0).astype(
                {f_csl_fid: np.int})
        df_intp     = df_intp.sort_values(f_csl_fid, axis=0).astype(
                {f_csl_fid: np.int, f_fa: np.int})
        df_intp_tgb = df_intp_tgb.sort_values(f_csl_fid, axis=0).astype(
                {f_csl_fid: np.int, f_fa: np.int, f_tgb: np.int})
        
        # control input data
        error_str = ''
        # CSL centered in a wrong type of model cell (e.g. head water cell)
        tgb_cs_un, tgb_cs_un_count = np.unique(df_intp_tgb.loc[:, f_tgb], return_counts=True)
        tgb_cs_val = np.isin(tgb_cs_un, ser_rout_tgb)
        error_wrong_tgb_typ = np.any(~tgb_cs_val)
        if error_wrong_tgb_typ:
            tgb_cs_not_val = tgb_cs_un[~tgb_cs_val]
            error_str = error_str + ('ERROR: There is/are {0:d} profile/s which '
                                     'is/are centered in a wrong model cell type. '
                                     'It could be a head water cell.\n'
                                     'TGB-ID(s):\n').format(tgb_cs_not_val.shape[0])
            for id_wr in tgb_cs_not_val:
                error_str = error_str + '   {0:d}\n'.format(int(id_wr))
        # CSL with wrong number of intersection points
        csl_un, csl_un_count = np.unique(df_intp_tgb.loc[:, f_csl_fid], return_counts=True)
        error_csl_wrong_intp = np.any(csl_un_count != 1)
        if error_csl_wrong_intp:
            csl_id_wrong = np.int64(csl_un[csl_un_count != 1])
            error_str = error_str + ('ERROR: There is/are {0:d} profile/s which '
                                     'has/have wrong number of intersection points. '
                                     'The distance of CSL-FNW intersection point '
                                     'to a model cell boarder could be less than 0.01m.\n'
                                     'CSL-ID(s):\n').format(csl_id_wrong.shape[0])
            for id_wr, nb_wr in zip(csl_id_wrong, csl_un_count[csl_un_count != 1]):
                error_str = error_str + '   {0:d} ({1:d} intersection points)\n'.format(id_wr, nb_wr)
        # profiles with no or wrong intersecting FNW element
        error_csl_nowrong_fnw = df_intp_tgb.shape[0] != np.unique(df_csp.loc[:, f_csl_fid]).shape[0]
        if error_csl_nowrong_fnw:
            csl_id_missing = np.int64(np.setdiff1d(np.unique(df_csp.loc[:,f_csl_fid]), df_intp_tgb.loc[:, f_csl_fid]))
            error_str = error_str + ('ERROR: There is/are {0:d} profile/s which '
                                     'is/are either not intersecting the flow network, '
                                     'or which does/do not intersect the river network '
                                     'at the intended element.\n'
                                     'CSL-ID(s):\n').format(csl_id_missing.shape[0])
            for id_mis in csl_id_missing:
                error_str = error_str + '   {0:d}\n'.format(id_mis)
        # multiple definition of cross sections for model cells
        tgb_doubles = tgb_cs_un[tgb_cs_un_count != 1]
        error_mult_csl_per_tgb = tgb_doubles.shape[0] != 0
        if error_mult_csl_per_tgb:
            error_str = error_str + ('ERROR: There is/are {0:d} model cell/s with '
                                     'more than one allocated cross sections.\n'
                                     'TGB-ID(s):\n'.format(tgb_doubles.shape[0]))
            for tgb_double in tgb_doubles:
                error_str = error_str + '   {0:d}\n'.format(int(tgb_double))
        # print error message and exit
        if error_csl_wrong_intp or error_csl_nowrong_fnw or error_mult_csl_per_tgb:
            print(error_str)
            sys.exit(0)    
        
        return df_csp, df_intp, df_intp_tgb
    
    # %% get user defined cross section profile elevation information
    def get_userdef_cs_dem_prof(df_csp, df_intp, df_intp_tgb,
                                def_ch_w=0.5, def_ch_h=0.5, def_ch_wres=0.05,
                                print_out=False):
        """
        This function extracts the elevation information of the cross sections
        and summarizes the information into pandas.Series. Additionally, the
        function applies the following corrections:
            1) If there are intersection points with other flow channels, than
               the cross section profile is clipped at the watershed between
               the mid-point of the cross section (= intersection point with
               channel) and the 'false' flow channel. 
            2) If the cross section profile is monotonically decreasing at 
               either the left or the right channel side, than the cross section
               profile at this side is replaced by a artificial channel defined
               by the user.
        Finally, also the unit vector of the cross section profile (x-/y-direction)
        is returned.
    
        JM 2021
    
        Arguments:
        -----------
        df_csp: pandas.DataFrame
            DataFrame containing all information for the cross section line:
            - csl_fid: cross section line feature ID
            - x, y, z: X-, Y-, and Z-coordinates for points along the cross section
        df_intp: pandas.DataFrame
            DataFrame containing all information for the cross sections' 
            intersection points with potential other flow channel around:
            - csl_fid: cross section line feature ID
            - x, y, z: X-, Y-, and Z-coordinates for mid-point
            - fa:      flow accumulation value at intersection point
        df_intp_tgb: pandas.DataFrame
            DataFrame containing all information for the cross sections' mid-point
            (is equal to the intersection point with the original flow channel):
            - csl_fid: cross section line feature ID
            - x, y, z: X-, Y-, and Z-coordinates for mid-point
            - tgb:     model element ID number
            - fa:      flow accumulation value at intersection point
        def_ch_w, def_ch_h: float (optional, default: 0.5, 0.5) [m]
            artificial channel width (w) and depth (h), added to continuiously
            descending cross sections
        def_ch_wres: float (optional, default: 0.05)
            horizontal resolution of interpolated points within channel
        print_out: boolean (optional, default: False)
            true if workprogress shall be print to command line
    
        Returns:
        -----------
        ser_cs_l, ser_cs_h: pandas.Series
            Series of cross sections' distance (ser_cs_l) and elevation
            difference (ser_cs_h) from flow network channel (mid-point).
        ser_cs_uvec: pandas.Series
            unit vectors of cross section lines
        """
        f_tgb = 'tgb'
        f_csl_fid = 'csl_fid'
        f_x = 'x'
        f_y = 'y'
        f_z = 'z'

        
        if print_out: print('...get user defined cross section profiles...')
        # pre-define series for interpolated cross section points
        ser_csl_tgb = df_intp_tgb.loc[:, f_tgb].sort_values()
        ser_cs_l    = pd.Series(df_intp_tgb.shape[0]*[[]],
                                index=ser_csl_tgb, name='cs_l')
        ser_cs_h    = pd.Series(df_intp_tgb.shape[0]*[[]],
                                index=ser_csl_tgb, name='cs_h')
        ser_cs_uvec = pd.Series(np.zeros((df_intp_tgb.shape[0], )) * np.nan, 
                                index=ser_csl_tgb, name='uvec')
        # iteratie cross sections
        for oid, (csl_id, _, _, _, fa, tgb) in df_intp_tgb.iterrows():
        
            # get cross section vertices' data
            csp_id_ii = df_csp.loc[:, f_csl_fid] == csl_id
            csp_x = df_csp.loc[csp_id_ii, f_x].to_numpy()
            csp_y = df_csp.loc[csp_id_ii, f_y].to_numpy()
            csp_h = df_csp.loc[csp_id_ii, f_z].to_numpy()
            # get cross section midpoint data
            intp_tgb_id_ii = df_intp_tgb.loc[:, f_csl_fid] == csl_id
            cs_intp_tgb_x = df_intp_tgb.loc[intp_tgb_id_ii, f_x].to_numpy()
            cs_intp_tgb_y = df_intp_tgb.loc[intp_tgb_id_ii, f_y].to_numpy()
            # get data of false intersection points
            intp_id_ii = df_intp.loc[:, f_csl_fid] == csl_id
            cs_intp_x = df_intp.loc[intp_id_ii, f_x].to_numpy()
            cs_intp_y = df_intp.loc[intp_id_ii, f_y].to_numpy()
            
            # calculate distances from low point
            # calculate coordinate differences between vertices and low point
            csp_dx = csp_x - cs_intp_tgb_x
            csp_dy = csp_y - cs_intp_tgb_y
            div_val_ii = np.logical_and(csp_dx != 0, csp_dy != 0)
            ser_cs_uvec.at[tgb] = np.mean(np.divide(
                    csp_dx[div_val_ii], csp_dy[div_val_ii]), axis=0)
            # calculate signed Euklidian vertex differences to point list
            if np.all(csp_dy == 0): dif_sign = np.sign(csp_dx)
            else:                   dif_sign = np.sign(csp_dy)
            csp_l = np.multiply(np.sqrt(csp_dx ** 2 + csp_dy ** 2), dif_sign)
            # sort rows in descending distance order
            csp_l_ii = (csp_l).argsort()
            cs_l_r = csp_l[csp_l_ii]
            # find position of intersection point with flow network
            cs_l_n_ii = np.argmin(np.abs(cs_l_r))
            # correct cross section points' elevation
            csp_h = csp_h[csp_l_ii]
            cs_h_r = csp_h - csp_h[cs_l_n_ii]
            # split cross section in two parts (left and right)
            cs_h_rl = cs_h_r[:cs_l_n_ii]
            cs_h_rr = cs_h_r[cs_l_n_ii:]
            cs_l_rl = cs_l_r[:cs_l_n_ii]
            cs_l_rr = cs_l_r[cs_l_n_ii:]
        
            # other stream in cross section -> need for watershed divide calculation
            if np.any(intp_id_ii):
                # calculate coordinate differences between intersection points and low point
                cs_intp_dx = cs_intp_x - cs_intp_tgb_x
                cs_intp_dy = cs_intp_y - cs_intp_tgb_y
                # calculate signed Euklidian intersection point differences
                if np.all(csp_dy == 0): dif_sign = np.sign(cs_intp_dx)
                else:                   dif_sign = np.sign(cs_intp_dy)
                cs_intp_l = np.multiply(np.sqrt(cs_intp_dx ** 2 + cs_intp_dy ** 2), dif_sign)
                # select closest intersection points with false channels for both sides
                cs_intp_ll = cs_intp_l[cs_intp_l < 0]
                cs_intp_lr = cs_intp_l[cs_intp_l > 0]
                # other stream intersection in left part of CS -> clip left CS part
                if cs_intp_ll.shape[0] != 0:
                    # sort rows in descending distance order
                    cs_intp_ll = cs_intp_ll[(-cs_intp_ll).argsort()]
                    # identify cross section profile until false channel
                    cs_fch_val_booll = cs_l_rl > cs_intp_ll[0]
                    # clip cross section profile
                    cs_l_rl = cs_l_rl[cs_fch_val_booll]
                    cs_h_rl = cs_h_rl[cs_fch_val_booll]
                # other stream in right part of CS -> clip right CS part
                if cs_intp_lr.shape[0] != 0:
                    # sort rows in ascending distance order
                    cs_intp_lr = cs_intp_lr[(cs_intp_lr).argsort()]
                    # identify cross section profile until false channel
                    cs_fch_val_boolr = cs_l_rr < cs_intp_lr[0]
                    # clip cross section profile
                    cs_l_rr = cs_l_rr[cs_fch_val_boolr]
                    cs_h_rr = cs_h_rr[cs_fch_val_boolr]
            # clip cross section at both sides' elevation maxima
            cs_h_maxl = np.argmax(cs_h_rl)
            cs_h_maxr = np.argmax(cs_h_rr)
            # merge cross section parts (left and right)
            cs_l_rc = np.hstack((cs_l_rl[cs_h_maxl:], cs_l_rr[:cs_h_maxr]))
            cs_h_rc = np.hstack((cs_h_rl[cs_h_maxl:], cs_h_rr[:cs_h_maxr]))
            # if cross section is continuatly descending at left side,
            # add default channel at left side
            if np.all(cs_l_rc <= 0):
                if cs_l_rc[-1] < 0: # if zero is not included in cross section
                    cs_l_rc_add = np.arange(0, def_ch_w, def_ch_wres)
                    cs_h_rc_add = np.arange(0, def_ch_w, def_ch_wres)                            
                else: # if zero is included in cross section
                    cs_l_rc_add = np.arange(def_ch_wres, def_ch_h, def_ch_wres)
                    cs_h_rc_add = np.arange(def_ch_wres, def_ch_h, def_ch_wres)                            
                cs_l_rc = np.hstack((cs_l_rc, cs_l_rc_add))
                cs_h_rc = np.hstack((cs_h_rc, cs_h_rc_add))
            # if cross section is continuatly descending at right side,
            # add default channel at the right side
            if np.all(cs_l_rc >= 0):
                if cs_l_rc[0] > 0: # if zero is not included in cross section
                    cs_l_rc_add = np.arange(0, def_ch_w, def_ch_wres)
                    cs_h_rc_add = np.arange(0, def_ch_w, def_ch_wres)                            
                else: # if zero is included in cross section
                    cs_l_rc_add = np.arange(def_ch_wres, def_ch_h, def_ch_wres)
                    cs_h_rc_add = np.arange(def_ch_wres, def_ch_h, def_ch_wres)                            
                cs_l_rc = np.hstack((cs_l_rc_add[::-1], cs_l_rc))
                cs_h_rc = np.hstack((cs_l_rc_add[::-1], cs_h_rc))
            # save to series
            ser_cs_l.at[tgb] = np.transpose(cs_l_rc, 0)
            ser_cs_h.at[tgb] = np.transpose(cs_h_rc, 0)
            
        return ser_cs_l, ser_cs_h, ser_cs_uvec
    
    # %% get lower channel bank coordinates and create GIS polyline feature class
    def calc_ch_bank_wsl(df_intp_tgb, ser_cs_uvec,
                         ser_wsll_ll, ser_wslr_ll, ser_ll_ii_fit, sr_obj,
                         path_gdb_out, name_ch_fit_coords='ch_fit_coords',
                         name_bcwsl='bcwsl', 
                         print_out=False):
        """
        This function extracts the lower channel bank x- and y-coordinates at
        both sides of the channel and creates a polyline feature class out of it.
        The coordinates are returned as well.
    
        JM 2021
    
        Arguments:
        -----------
        df_intp_tgb: pandas.DataFrame
            DataFrame containing all information for the cross sections' mid-point
            (is equal to the intersection point with the original flow channel):
            - csl_fid: cross section line feature ID
            - x, y:    X- and Y-coordinates for mid-point
            - tgb:     model element ID number
        ser_cs_uvec: pandas.Series
            unit vectors of cross section lines
        ser_wsll_ll, ser_wslm_ll, ser_wslr_ll: pandas.Series
            Series containing the left, mean, and right water surface levels
            for each lamella defined.
        sr_obj: arcpy.SpatialReferenceObject
            arcpy.Object containing the spatial reference of the final feature class
        path_gdb_out: str (e.g., 'c:\model_creation.gdb')
            path of the output file geodatabase
        name_ch_fit_coords: str (optional, default: 'ch_fit_coords')
            name of coordinate table for bankful discharge line
        name_bcwsl: str (optional, default: 'bcwsl')
            name of bankful channel water surface line
        print_out: boolean (optional, default: False)
            true if workprogress shall be print to command line        
        
        Returns:
        -----------
        df_ch_fit_coords: pandas.DataFrame
            Dataframe containing the X- and Y- coordinates of the points limiting
            the left (L_x, l_y) and right (r_x, r_y) bankful channel line
        """
        if print_out: print('...calculate lower channel bank water surface line...')
        # (de-)activate additional debugging command line output
        debug = False # (False/True)
        # define internal field names
        f_tgb     = 'tgb'
        f_csl_fid = 'csl_fid'
        f_x       = 'x'
        f_y       = 'y'
        f_ch_l_x  = 'l_x'
        f_ch_l_y  = 'l_y'
        f_ch_r_x  = 'r_x'
        f_ch_r_y  = 'r_y'
        # set workspace
        arcpy.env.workspace = path_gdb_out
        # coordinate arrays of bankfull discharge line
        df_ch_fit_coords = pd.DataFrame(np.zeros((df_intp_tgb.shape[0], 4)) * np.nan,
                                        index=df_intp_tgb.loc[:, f_tgb],
                                        columns=[f_ch_l_x, f_ch_l_y, f_ch_r_x, f_ch_r_y])
        # set itnersection points index to model element index
        df_intp_tgb.index = df_intp_tgb.loc[:, f_tgb]
        # iterate model elements' mean cross sections
        for tgb, (uvec, wsll_ll, wslr_ll, ll_ii, ch_x, ch_y) \
            in pd.concat((ser_cs_uvec, ser_wsll_ll, ser_wslr_ll, ser_ll_ii_fit,
                          df_intp_tgb.loc[:, [f_x, f_y]]), axis=1).iterrows():
            # if fit was successfull, calculate coordinates
            if not np.isnan(ll_ii):
                ll_ii = np.int(ll_ii)
                if print_out and debug:
                    print(('{0:4d}: {1:6.2f}, h: {2:4.1f}, w: {3:6.2f} ({4:6.2f}, '
                           '{5:6.2f}), ({6:.2f}, {7:.2f})').format(
                            tgb, uvec, h_ll[ll_ii], wsll_ll[ll_ii] + wslr_ll[ll_ii],
                            wsll_ll[ll_ii], wslr_ll[ll_ii], ch_x, ch_y))
                # get coordinates of left side
                dyl = wsll_ll[ll_ii] / (uvec ** 2 + 1)
                df_ch_fit_coords.at[tgb, f_ch_l_y] = ch_y - dyl
                df_ch_fit_coords.at[tgb, f_ch_l_x] = ch_x - dyl * uvec
                # get coordinates of right side
                dyr = wslr_ll[ll_ii] / (uvec ** 2 + 1)
                df_ch_fit_coords.at[tgb, f_ch_r_y] = ch_y + dyr
                df_ch_fit_coords.at[tgb, f_ch_r_x] = ch_x + dyr * uvec
        # export coordinate table
        df_ch_fit_coords_tab = pd.concat([df_intp_tgb.loc[:, [f_tgb, f_csl_fid]],
                                          df_ch_fit_coords], axis=1).sort_index()
        tc.df_to_table(df_ch_fit_coords_tab, path_gdb_out, name_ch_fit_coords)
        # define paths
        path_ch_fit_coords = path_gdb_out + name_ch_fit_coords
        path_bcwsl         = path_gdb_out + name_bcwsl
        # create bankful water surface indication line
        arcpy.XYToLine_management(
                path_ch_fit_coords, path_bcwsl, f_ch_l_x, f_ch_l_y, f_ch_r_x, f_ch_r_y,
                line_type='GEODESIC', id_field=f_csl_fid, spatial_reference=sr_obj)
        
        return df_ch_fit_coords
    
    # %% correct cross section area per lamella
    def corr_a_ll(h_ll, ser_wslm_ll, ser_a_ll, ser_p_ll, ser_ll_ii_fit, ser_csl_fid,
                  def_profdat_decmax=2, print_out=False):
        """
        This function corrects mismatches of the Series containing the cross 
        section area by lamella. Because of the discretized profile and the way
        how depressions are involved in the calculation, the difference between
        to ascending lamellae might get negative. This is corrected with this 
        function returning a corrected Series.
    
        JM 2021
    
        Arguments:
        -----------
        h_ll: np.array (int)
            Numpy array of lamellae used to describe the profiles
        ser_wslm_ll: pandas.Series
            Series containing the mean water surface levels for each lamella 
            defined.
        ser_a_ll, ser_p_ll: pandas.Series
            Series containing the cross section area and wetted perimeter 
            for each lamella defined.
        ser_ll_ii_fit: pandas.Series (int)
            Series containing the lamella index for each model element found
            during the fitting process.
        ser_csl_fid: pandas.Series
            Series containing the cross section ID for all cross sections
        print_out: boolean (optional, default: False)
            true if workprogress shall be print to command line        
        
        Returns:
        -----------
        ser_a_ll_corr: pandas.Series
            Series containing the corrected cross section area for each lamella
            defined.
        """        
        if print_out: print('...correct cross section area per lamella...')
        # create a copy of cross section area per lamella
        ser_a_ll_corr = copy.deepcopy(ser_a_ll)
        # iterate cross section parameters
        for tgb, (wsl_ll, a_ll, ll_ii, csl_id) \
            in pd.concat((ser_wslm_ll, ser_a_ll, ser_ll_ii_fit,
                          ser_csl_fid), axis=1).iterrows():
            # calculate area difference within main channel per lamella
            da_dwsl = np.round(wsl_ll[ll_ii], def_profdat_decmax) * (h_ll - h_ll[ll_ii])
            # calculate cross section area difference per lamella
            da = np.round(a_ll, def_profdat_decmax) - np.round(a_ll[ll_ii], def_profdat_decmax)
            # calculate area difference in foreland per lamella
            da_fl = da - da_dwsl
            # set foreland area to zero for lamellae lower bankful discharge level
            da_fl[h_ll <= h_ll[ll_ii]] = 0
            # find lamellae, where foreland area is supposed to be < 0
            da_fl_neg_ii = da_fl < 0
            # if there is any negative foreland area lamella
            if np.any(da_fl_neg_ii):
                # add negative differences to cross section area
                da_fl_neg   = da_fl[da_fl_neg_ii] * -1
                da_fl_neg_r = np.round(da_fl_neg, def_profdat_decmax)
                da_fl_neg[da_fl_neg <  da_fl_neg_r] = da_fl_neg_r[da_fl_neg <  da_fl_neg_r]
                da_fl_neg[da_fl_neg >= da_fl_neg_r] = da_fl_neg_r[da_fl_neg >= da_fl_neg_r] \
                                                      + 10 ** -def_profdat_decmax
                a_ll[da_fl_neg_ii] = a_ll[da_fl_neg_ii] + da_fl_neg
                if print_out: 
                    print(('   Profile ID {0:.0f} (TGB: {1:.0f}): '
                           'The following lamella(s) have been corrected:').format(
                            csl_id, tgb))
                    for ll, a_corr in zip(h_ll[da_fl_neg_ii], da_fl_neg):
                        print('      {0:.1f}m: +{1:.2f} m²'.format(ll, a_corr))
            # save corrected W-A-relation
            ser_a_ll_corr.at[tgb] = a_ll
        return ser_a_ll_corr    
    
    # %% get bankful channel parameters
    def get_bc_par(h_ll, ser_wslm_ll, ser_a_ll_corr, ser_p_ll, ser_ll_ii_fit,
                   ser_tgb_id, print_out=False):
        
        """
        This function summarizes the parameters water surface level hight (h) 
        and width (wsl), cross section area (a), and wetted perimeter (p) for
        the bankful channel flow.
    
        JM 2021
    
        Arguments:
        -----------
        h_ll: np.array (int)
            Numpy array of lamellae used to describe the profiles
        ser_wslm_ll: pandas.Series
            Series containing the mean water surface levels
            for each lamella defined.
        ser_a_ll_corr, ser_p_ll: pandas.Series
            Series containing the corrected cross section area and wetted
            perimeter for each lamella defined.
        ser_ll_ii_fit: pandas.Series (int)
            Series containing the lamella index for each model element found
            during the fitting process.
        ser_tgb_id: pandas.Series
            Series containing the model element ID for all cross sections
        print_out: boolean (optional, default: False)
            true if workprogress shall be print to command line        
        
        Returns:
        -----------
        df_ch_fit_par: pandas.DataFrame
            DataFrame containing the water level height (h), cross section
            area (a), wetted perimeter (p) and water surface level width (wsl)
            for the bankful channel.
        """
        if print_out: print('...get bankful channel parameters...')
        # define internal field names
        f_h   = 'h'
        f_a   = 'a'
        f_p   = 'p'
        f_wsl = 'wsl'
        # pre-define DataFrame
        df_ch_fit_par = pd.DataFrame(np.zeros((ser_tgb_id.shape[0], 4)) * np.nan,
                                  index=ser_tgb_id,
                                  columns=[f_h, f_a, f_p, f_wsl])
        # iterate cross section parameters
        for tgb, (wsl_ll, a_ll, p_ll, ll_ii) \
            in pd.concat((ser_wslm_ll, ser_a_ll_corr, ser_p_ll, ser_ll_ii_fit),
                         axis=1).iterrows():
            # save bankful channel characteristics
            df_ch_fit_par.at[tgb, f_h  ] =   h_ll[ll_ii]
            df_ch_fit_par.at[tgb, f_a  ] =   a_ll[ll_ii]
            df_ch_fit_par.at[tgb, f_p  ] =   p_ll[ll_ii]
            df_ch_fit_par.at[tgb, f_wsl] = wsl_ll[ll_ii]
        return df_ch_fit_par
    
    # %% allocate cross section IDs to model cell IDs
    def all_cs_to_tgb(ser_csl_fid, ser_tgb_down, ser_tgb_down_nd, 
                      ser_tgb_type_routing, print_out=False):
        """
        This function allocates cross section IDs to model IDs. The calculation
        process pre-defines, that not every routing element is getting assigned a
        cross section line. Therefore, an upstream cross section is allocated to 
        'empty' downstream routing elements as well. Dummy elements are ignored.
        The function returns a Series with allocated cross section IDs.
        
        JM 2021
    
        Arguments:
        -----------
        ser_csl_fid: pandas.Series
            Series containing the cross section ID for all cross sections
        ser_tgb_down: pandas.Series
            Series of downstream model element indices corresponding to the serie's
            ascending index. The last value is outlet, identified with a zero.
            The outlet will be neglected in calculations.
            (e.g., pd.Series([2, 4, 4, 0], index=[1, 2, 3, 4], name='tgb_down'))
        ser_tgb_down_nd: pandas.Series
            Series of corresponding downstream model element indices ignoring dummy
            elements. Model outlet remains -1 and dummy elements are represented as 0.
        ser_tgb_type_routing: pandas.Series
            Boolean Series, which identifies the routing cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[0, 0, 1, 1], index=[1, 2, 3, 4],
                             name='routing', dtype='bool'))
        print_out: boolean (optional, default: False)
            true if workprogress shall be print to command line        
        
        Returns:
        -----------
        ser_tgb_csl: pandas.Series
            Series of cross section ID numbers, which are allocated to all 
            routing model elements in the model structure
        """
        if print_out: print('...allocate cross sections to TGB...')
        # (de-)activate additional debugging command line output
        debug = True # (False/True)
        # get index
        ser_csl_fid = ser_csl_fid.sort_index()
        # get downstream index
        ser_csl_fid_tgb = pd.Series(ser_csl_fid.index.values,
                                    index=ser_csl_fid.index).shift(-1)
        # get attributes of outflow
        tgb_out = np.max(ser_tgb_down.index)
        tgb_down_out = ser_tgb_down.at[tgb_out]
        # pre-allocate Series
        ser_tgb_csl = pd.Series(np.zeros((ser_tgb_down.shape[0], )) * np.nan,
                                index=ser_tgb_down.index, name=ser_csl_fid.name)
        ser_tgb_csl_pre = pd.Series(np.zeros((ser_tgb_down.shape[0], )) * np.nan,
                                    index=ser_tgb_down.index, name=ser_csl_fid.name)
        # get model element ID of last defined cross section
        tgb_last_csl = np.max(ser_csl_fid.index)
        # iterate cross section IDs
        for tgb, csl in ser_csl_fid.iteritems():
            if debug and print_out:
                print('set tgb {0:5d} with cs_fid {1:5d}'.format(tgb, csl))
            ser_tgb_csl.at[tgb] = csl
            if ser_tgb_down.at[tgb] == tgb_down_out:
                break
            # get index of real downstream model cell
            tgb_down = ser_tgb_down_nd.at[tgb]
            # if it is not the last cross section
            if tgb != tgb_last_csl:
                # save reference element ID
                tgb_ref = copy.deepcopy(tgb)
                # use recent cross section for all downstream cells (= higher ID) 
                # until next cross section's model cell in list
                while ser_csl_fid_tgb.loc[tgb_ref] > tgb_down:
                    # set downstream cell as recent cell
                    tgb = copy.deepcopy(tgb_down)
                    # if there is already pre-set a cross section ID, use this ID
                    # as it represents the largest upstream area following from 
                    # model structure algorithm (ID = 1 for largest flow length to outlet)
                    if not np.isnan(ser_tgb_csl_pre[tgb]):
                        csl = copy.deepcopy(ser_tgb_csl_pre[tgb])
                        if debug and print_out:
                            print('set tgb {0:5d} with cs_fid {1:5d} (pre-set)'.format(
                                    tgb, int(csl)))
                    elif ser_tgb_type_routing.at[tgb]:
                        if debug and print_out:
                            print('set tgb {0:5d} with cs_fid {1:5d}'.format(
                                    tgb, int(csl)))
                    # if it is a routing cell, set cross section and model cell IDs
                    # for recent model cell
                    if ser_tgb_type_routing.at[tgb_down]:
                        ser_tgb_csl.at[tgb] = csl
                    # if downstream cell identifies recent cell as outlet or 
                    # real dummy cell partner is upstream (=outlet of sub-catchment),
                    # break loop
                    if ser_tgb_down.at[tgb] == tgb_down_out:
                        break
                    # get index of real downstream model cell
                    tgb_down = ser_tgb_down_nd.at[tgb]
                    if np.isnan(tgb_down):
                        break
                # if there is not already a pre-set and it is a routing cell,
                # pre-set cross section ID for downstream cell
                if not np.isnan(tgb_down):
                    if np.isnan(ser_tgb_csl_pre[tgb_down]):
                        ser_tgb_csl_pre[tgb] = csl
                        if debug and print_out: 
                            print('pre-set tgb_down {0:5d} with cs_fid {1:5d}'.format(
                                    tgb_down, int(csl)))
            # if it is the last cross section
            else:
                # use recent cross section for all downstream cells (= higher ID) 
                # until (and inclusive) outlet
                while tgb_out >= tgb_down:
                    # set downstream cell as recent cell
                    tgb = copy.deepcopy(tgb_down)
                    # if there is already pre-set a cross section ID, use this ID
                    # as it represents the largest upstream area following from 
                    # model structure algorithm (ID = 1 for largest flow length to outlet)
                    if not np.isnan(ser_tgb_csl_pre[tgb]):
                        csl = copy.deepcopy(ser_tgb_csl_pre[tgb])
                        if debug and print_out: 
                            print('last: set tgb {0:5d} with cs_fid {1:5d} (pre-set)'.format(
                                    tgb, int(csl)))
                    elif ser_tgb_type_routing.at[tgb]:
                        if debug and print_out: 
                            print('last: set tgb {0:5d} with cs_fid {1:5d}'.format(
                                    tgb, int(csl)))
                    # if it is a routing cell, set cross section and model cell IDs
                    # for recent model cell
                    if ser_tgb_type_routing.at[tgb_down]:
                        ser_tgb_csl.at[tgb] = csl
                    # if downstream cell identifies recent cell as outlet, break loop
                    if ser_tgb_down.at[tgb] == tgb_down_out:
                        break
                    # get index of real downstream model cell
                    tgb_down = ser_tgb_down_nd.at[tgb]
                    if np.isnan(tgb_down):
                        break
        # reduce list to cell IDs with connected cross section
        ser_tgb_csl = ser_tgb_csl.loc[~np.isnan(ser_tgb_csl)].astype(np.int)
        
        return ser_tgb_csl
    
    # %% calculations
    # define internal variable names
    f_tgb = 'tgb'
    f_csl_fid = 'csl_fid'
    # set workspace
    arcpy.env.workspace = path_gdb_out
    # get cross section GIS data information
    ser_rout_tgb = ser_tgb_type_routing.index[ser_tgb_type_routing].to_numpy()
    df_csp, df_intp, df_intp_tgb = get_cs_gisdata(
            path_fnw, path_fnw_o, path_csl, path_dem_hr, path_fa_hr, 
            path_tgb_s, ser_rout_tgb,
            path_gdb_out, def_cs_wmax_eval=def_cs_wmax_eval,
            def_cs_intp_buf_dist=def_cs_intp_buf_dist,
            print_out=print_out)
    # get user defined cross section profile elevation information
    ser_cs_l, ser_cs_h, ser_cs_uvec = get_userdef_cs_dem_prof(
            df_csp, df_intp, df_intp_tgb,
            def_ch_w=def_ch_w, def_ch_h=def_ch_h, def_ch_wres=def_ch_wres,
            print_out=print_out)
    # estimate cross section water surface levels, cross section area and
    # wetted perimeter for each lamella
    h_ll, ser_wsll_ll, ser_wslm_ll, ser_wslr_ll, df_ch_h, ser_a_ll, ser_p_ll = \
        tc.est_ch_wsl(ser_cs_l, ser_cs_h, ser_ch_form_q,
                   def_cs_hmax_eval=def_cs_hmax_eval, def_lam_hres=def_lam_hres,
                   def_ch_vmin=def_ch_vmin, def_ch_vmax=def_ch_vmax)
    # fit channel depth and width
    ser_ch_h_fit, df_ch_wsl_fit, ser_ll_ii_fit = tc.fit_ch(
            ser_pef_bm, ser_pef_hm, ser_cs_l, ser_cs_h, 
            ser_wsll_ll, ser_wslm_ll, ser_wslr_ll, 
            def_ch_wmax_eval=def_ch_wmax_eval, def_lam_hres=def_lam_hres,
            def_chbank_slmin=def_chbank_slmin, def_ch_hmin=def_ch_hmin,
            def_ch_hmin_eval=def_ch_hmin_eval, 
            ctrl_show_plots=ctrl_show_plots, ctrl_save_plots=ctrl_save_plots, 
            ser_tgb_a=ser_ft, ser_tgb_a_in=ser_area_outfl, ser_tgb_q_in=ser_ch_form_q, 
            def_ch_hmax_eval=def_ch_hmax_eval, path_plots_out=path_plots_out)
    # get lower channel bank values for GIS polyline creation
    sr_obj = arcpy.Describe(path_tgb_s).spatialReference
    _ = calc_ch_bank_wsl(
            df_intp_tgb, ser_cs_uvec, ser_wsll_ll, ser_wslr_ll, ser_ll_ii_fit, sr_obj,
            path_gdb_out, name_ch_fit_coords=name_ch_fit_coords, name_bcwsl=name_bcwsl,
            print_out=print_out)
    # correct cross section area per lamella
    ser_csl_fid = pd.Series(df_intp_tgb.loc[:, f_csl_fid], index=df_intp_tgb.index)
    ser_a_ll_corr = corr_a_ll(h_ll, ser_wslm_ll, ser_a_ll, ser_p_ll, ser_ll_ii_fit,
                              ser_csl_fid, def_profdat_decmax=def_profdat_decmax,
                              print_out=print_out)
    # get bankful channel parameters
    ser_tgb_id = df_intp_tgb.loc[:, f_tgb]
    df_ch_fit_par = get_bc_par(h_ll, ser_wslm_ll, ser_a_ll_corr, ser_p_ll, 
                               ser_ll_ii_fit, ser_tgb_id, print_out=print_out)
    # export parameters to table
    df_profpar = pd.concat([ser_tgb_id, ser_csl_fid, df_ch_fit_par], axis=1).sort_index()
    tc.df_to_table(df_profpar, path_gdb_out, name_profile_par)
    # allocate cross section IDs to model cell IDs
    ser_tgb_csl = all_cs_to_tgb(ser_csl_fid, ser_tgb_down, ser_tgb_down_nd,
                                ser_tgb_type_routing, print_out=print_out)
    # concatenate data to one DataFrame
    df_profdat_par = pd.concat([ser_csl_fid, df_ch_fit_par, ser_ll_ii_fit, 
                                ser_a_ll, ser_p_ll, ser_wslm_ll],
                               axis=1).sort_index()

    return df_profdat_par, ser_tgb_csl