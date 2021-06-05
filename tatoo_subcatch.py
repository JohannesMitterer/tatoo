# -*- coding: utf-8 -*-
"""
This library contains all functions needed to produce the spatial files
of a LARSIM subcatchment model (tgb.dat, utgb.dat). It uses functions from
the TATOO core library.

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
import lmfit
import arcpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import (InsetPosition, mark_inset)
import tatoo_common as tc

# check out ArcGIS spatial analyst license
class LicenseError(Exception):
    pass
try:
    if arcpy.CheckExtension('Spatial') == 'Available':
        arcpy.CheckOutExtension('Spatial')
        print ('Checked out \'Spatial\' Extension')
    else:
        raise LicenseError
except LicenseError:
    print('Spatial Analyst license is unavailable')
except:
    print(arcpy.GetMessages(2))

# allow overwriting the outputs
arcpy.env.overwriteOutput = True


# %% calculate subcatchments optimizing pour points according to specifications
def optimize_subc(path_dem, path_fnw, path_pp_ws, field_pp_ws,
                  def_sc_area, def_fl_min_tol, path_gdb_out, path_files_out, 
                  name_dem_c='dem_c', name_fd_c='fd_c', name_fa_c='fa_c', 
                  name_fl_c='fl_c', name_ws_s='ws_s', name_pp_sc='pp_sc', 
                  name_sc_ws_s='sc_ws_s', name_fnw_fa='fnw_fa',
                  h_burn=10, print_out=False):
    """
    This function optimizes the location of subcatchment pour points within the
    model domain to match the specified subcatchment size. The digital elevation
    model is pre-processed using a polyline feature class representing the flow
    network to ensure flux along existing rivers (ignoring errors of the digital
    elevation model). Additionally, important parameters as the pour points'
    coordinates (based on the elevation model's coordinates), the flow accumulation,
    and flow length at every point, as well as the downstream neighbours are 
    calculated and stored in the output. The user might check the shape of the
    resulting subwatersheds based on the additional resulting subwatershed polygons 
    and the flow network based on the flow accumulation.

    JM 2021

    Arguments:
    -----------
    path_dem: str
        path of the input digital elevation model raster
        (e.g., 'c:\model_creation.gdb\dem')
    path_fnw: str
        path of the flow network feature class or shape file
        (e.g., 'c:\model_creation.gdb\fnw.shp')
    path_pp_ws: str
        path of the model watershed pour point feature class
        (e.g., 'c:\model_creation.gdb\pp_ws')
    field_pp_ws: str
        name of the field in path_pp_ws containing the resulting watershed ID numbers
        and negative numbers for watersheds to be excluded
        (e.g., 'ModelWatershed')
    def_sc_area: integer
        intended size of the resulting optimized subwatershed model elements
        in [m2] (e.g., 10**6)
        attention: From its construction, the algorithm tends to underestimate
        the element size and gives in the mean 15% smaller model elements due to
        tendencially very small elements at confluence points
    def_fl_min_tol: integer
        tolerated negligible flow length in [m] (e.g., 50) in order to nibble
        very small subcatchments
    path_gdb_out: str
        path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
    path_files_out: str
        storage path for intermediate data (e.g., 'c:\tmp_model_data\')
    name_dem_c: str (optional, default: 'dem_c')
        name of the digital elevation raster within model domain
    name_fd_c: str (optional, default: 'fd_c')
        name of the flow direction raster within model domain
    name_fa_c: str (optional, default: 'fa_c')
        name of the flow accumulation raster within model domain
    name_fl_c: str (optional, default: 'fl_c')
        name of the flow length raster within model domain
    name_ws_s: str (optional, default: 'ws_s')
        name of the model domain watershed feature class
    name_pp_sc: str (optional, default: 'pp_sc')
        name of the subcatchment pour point feature class
    name_sc_ws_s: str (optional, default: 'sc_ws_s')
        name of the subcatchment polygon feature class
    name_fnw_fa: str (optional, default: 'fnw_fa')
        name of the flow network resulting from flow accumulation
    h_burn: integer (optional, default: 10)
        depth of river network burning in digital elevation model in [m]
    print_out: boolean (optional, default: False)
        true if workprogress shall be print to command line (default: false)

    Returns:
    -----------
    pp_points_df: pandas.DataFrame
        DataFrame of all parameters defining the resulting optimized pour points.
        The DataFrame includes the pour point ID as index and the following columns:
        - 'x': pour points' x-coordinate in the DEM coordinate system (float)
        - 'y': pour points' y-coordinate in the DEM coordinate system (float)
        - 'pp': pour point identification number (int)
        - 'pp_down': downstream pour point identification number (int)
        - 'fa': flow accumulation at pour points' location [-] (int)
        - 'fl': flow length at pour points' location [m] (float)
    """
    # pre-process gis data and calculate model watershed
    def preprocess_gisdata(path_dem, path_fnw, path_pp_ws, field_pp_ws,
                           path_gdb_out, name_dem_c='dem_c', name_fd_c='fd_c', 
                           name_fa_c='fa_c', name_fl_c='fl_c', name_ws_s='ws_s', 
                           h_burn=10,
                           print_out=False):
        """
        This function performs the necessary GIS-based pre-calculations for the
        pour point optimization process. Processing steps:
            1. convert flow network to raster
            2. imprint flow network in digital elevation model using h_burn
            3. fill imprinted digital elevation model
            4. calculate watershed
            5. extract DEM rasters and FD by watershed polygon
            6. calculate flow accumulation
            7. calculate flow length
    
        JM 2021
    
        Arguments:
        -----------
        path_dem: str
            path of the input digital elevation model raster
            (e.g., 'c:\model_creation.gdb\dem')
        path_fnw: str
            path of the flow network feature class or shape file
            (e.g., 'c:\model_creation.gdb\fnw.shp')
        path_pp_ws: str
            path of the model watershed pour point feature class
            (e.g., 'c:\model_creation.gdb\pp_ws')
        field_pp_ws: str
            name of the field in path_pp_ws containing the resulting watershed
            ID numbers (e.g., 'pointid')
        path_gdb_out: str
            path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
        name_dem_c: str (optional, default: 'dem_c')
            name of the digital elevation raster within model domain
        name_fd_c: str (optional, default: 'fd_c')
            name of the flow direction raster within model domain
        name_fa_c: str (optional, default: 'fa_c')
            name of the flow accumulation raster within model domain
        name_fl_c: str (optional, default: 'fl_c')
            name of the flow length raster within model domain
        name_ws_s: str (optional, default: 'ws_s')
            name of the model domain watershed feature class
        h_burn: integer (optional, default: 10)
            depth of river network burning in digital elevation model in [m]
        print_out: boolean (optional, default: False)
            true if workprogress shall be print to command line (default: false)
    
        Returns:
        -----------
        The function saves the resulting elevation, flow direction, flow
        accumulation, and flow length raster, as well as the model domain
        watershed in the defined file geodatabase.
        """
        # arcpy field names (ONLY CHANGE IF ARCPY FUNCTION CHANGES FORCE YOU!)
        f_oid = 'OBJECTID'
        f_val = 'Value'
        # define paths for outputs
        path_dem_c       = path_gdb_out + name_dem_c
        path_fd_c        = path_gdb_out + name_fd_c
        path_fa_c        = path_gdb_out + name_fa_c
        path_fl_c        = path_gdb_out + name_fl_c
        path_ws_s        = path_gdb_out + name_ws_s
        # define paths of intermediates in output geodatabase
        path_fd          = path_gdb_out + 'fd'
        path_ws_r        = path_gdb_out + 'ws_r'
        path_ws_sr       = path_gdb_out + 'ws_sr'
        path_fnw_r       = path_gdb_out + 'fnw_r'
        path_dem_cfnw    = path_gdb_out + 'dem_cfnw'
        path_dem_cfnw_f  = path_gdb_out + 'dem_cfnw_f'
        # set environments
        arcpy.env.workspace  = path_gdb_out
        arcpy.env.snapRaster = path_dem
        arcpy.env.extent     = path_dem
        # convert flow network to raster
        if print_out: print('...convert flow network to raster...')
        arcpy.conversion.PolylineToRaster(path_fnw, f_oid, path_fnw_r,
                                          'MAXIMUM_LENGTH', 'NONE', 1)
        # imprint flow network in DEM
        if print_out: print('...imprint flow network in DEM...')
        dem      = arcpy.Raster(path_dem)
        dem_cfnw = arcpy.sa.Con(arcpy.sa.IsNull(path_fnw_r), dem, dem - h_burn)
        dem_cfnw.save(path_dem_cfnw)
        # fill imprinted DEM
        if print_out: print('...fill imprinted digital elevation model...')
        dem_cfnw_f = arcpy.sa.Fill(path_dem_cfnw)
        dem_cfnw_f.save(path_dem_cfnw_f)
        # calculate flow direction
        if print_out: print('...calculate flow direction...')
        fd = arcpy.sa.FlowDirection(path_dem_cfnw_f)
        fd.save(path_fd)
        # calculate watershed
        if print_out: print('...calculate watershed...')
        arcpy.env.extent = 'MAXOF'
        ws_r = arcpy.sa.Watershed(path_fd, path_pp_ws, field_pp_ws)
        ws_r.save(path_ws_r)
        arcpy.conversion.RasterToPolygon(path_ws_r, path_ws_sr, 'NO_SIMPLIFY',
                                         f_val, 'SINGLE_OUTER_PART', None)
        # remove inflowing marked watersheds
        if print_out: print('...select model watershed...')
        pp_fieldnames = [field.name for field in arcpy.ListFields(path_pp_ws)]
        # if field exists, that identifies polygon as model watershed, delete watersheds
        # with the fields' value >= 1
        if field_pp_ws in pp_fieldnames:
            # join created watershed polygons to pour points
            path_ws_s_sj = path_gdb_out + 'ws_s_sj'
            arcpy.SpatialJoin_analysis(path_ws_sr, path_pp_ws, path_ws_s_sj,
                                       'JOIN_ONE_TO_ONE', 'KEEP_ALL',
                                       "{0:s} '{0:s}' true true false 2 Short 0 0,\
                                       First,#,pp,{0:s},-1,-1".format(field_pp_ws),
                                       'CONTAINS', '', '')
            # select and copy model watersheds marked with 1 (default value)
            sel_sql = field_pp_ws + ' >= 1'
            path_ws_s_sj_sel = arcpy.management.SelectLayerByAttribute(
                    path_ws_s_sj, 'NEW_SELECTION', sel_sql)
            path_ws_s = path_gdb_out + name_ws_s
            if arcpy.Exists(path_ws_s): arcpy.management.Delete(path_ws_s)
            arcpy.management.CopyFeatures(path_ws_s_sj_sel, path_ws_s,
                                          '', '', '', '')
            path_ws_s_sj_sel = arcpy.management.SelectLayerByAttribute(
                    path_ws_s_sj, 'CLEAR_SELECTION', sel_sql)
        else:
            if arcpy.Exists(path_ws_s): arcpy.management.Delete(path_ws_s)
            arcpy.management.CopyFeatures(path_ws_sr, path_ws_s,
                                          '', '', '', '')
        # extract DEM rasters and FD by watershed polygon
        if print_out: print('...extract by watershed polygon...')
        arcpy.env.extent = path_ws_s
        dem_c = arcpy.sa.ExtractByMask(dem, path_ws_s)
        dem_c.save(path_dem_c)
        fd_c  = arcpy.sa.ExtractByMask(path_fd, path_ws_s)
        fd_c.save(path_fd_c)
        # calculate flow accumulation
        if print_out: print('...calculate flow accumulation...')
        fa_c  = arcpy.sa.FlowAccumulation(path_fd_c)
        fa_c.save(path_fa_c)
        # calculate flow length
        if print_out: print('...calculate flow length...')
        fl_c  = arcpy.sa.FlowLength(path_fd_c, 'DOWNSTREAM', '')
        fl_c.save(path_fl_c)
        # release handles locking geodatabase
        dem        = None
        dem_c      = None
        dem_cfnw   = None
        dem_cfnw_f = None
        fd         = None
        fd_c       = None
        fa_c       = None
        fl_c       = None
        ws_r       = None
    
    # calculations
    # set workspace
    arcpy.env.workspace = path_gdb_out
    # define internal field names
    f_x       = 'x'
    f_y       = 'y'
    f_row     = 'row'
    f_col     = 'col'
    f_pp      = 'pp'
    f_dx      = 'dx'
    f_dy      = 'dy'
    f_fa      = 'fa'
    f_fl      = 'fl'
    f_fl_down = 'fl_down'
    f_jt_down = 'jt_down'
    f_pp_down = 'pp_down'
    f_jt_up   = 'jt_up'
    # arcpy field names (ONLY CHANGE IF ARCPY FUNCTION CHANGES FORCE YOU!)
    f_val     = 'Value'
    # define paths of intermediates in output geodatabase
    name_fa_dif      = 'fa_dif'
    path_ssp_sc      = path_gdb_out + 'ssp_sc'
    path_sc_ws_r     = path_gdb_out + 'sc_ws_r'
    path_fnw_fa_r    = path_gdb_out + 'fnw_fa_r'
    # define paths of intermediates in working folder
    path_fd_c_tif    = path_files_out + 'fd_c.tif'
    path_fa_c_tif    = path_files_out + 'fa_c.tif'
    path_fl_c_tif    = path_files_out + 'fl_c.tif'
    # (de-)activate debugging command line output
    debug = False # (False/True)
    
    # pre-process gis data and calculate model watershed
    preprocess_gisdata(path_dem, path_fnw, path_pp_ws, field_pp_ws,
                       path_gdb_out, name_dem_c=name_dem_c, name_fd_c=name_fd_c, 
                       name_fa_c=name_fa_c, name_fl_c=name_fl_c,
                       name_ws_s=name_ws_s, 
                       h_burn=h_burn,
                       print_out=True)
    
    # import flow direction, accumulation and length as numpy rasters
    fd, ncols, nrows, cellsz, xll, yll, ctrl_tif_export = tc.fdal_raster_to_numpy(
            path_gdb_out + name_fd_c, 'fd', path_fd_c_tif, True)
    fa, _, _, _, _, _, _ = tc.fdal_raster_to_numpy(
            path_gdb_out + name_fa_c, 'fa', path_fa_c_tif, False)
    fl, _, _, _, _, _, _ = tc.fdal_raster_to_numpy(
            path_gdb_out + name_fl_c, 'fl', path_fl_c_tif, True)
    
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
    # set flow accumulation thresholds
    fa_thr_sc_area = def_sc_area / cellsz
    fa_min_thr_sc_area = fa_thr_sc_area * 0.5
    # set flow length threshold
    fl_min_tol = def_fl_min_tol/cellsz
    # set arrays to nan if threshold is not met
    with np.errstate(invalid='ignore'):
        fa_l_thr_ii = fa < fa_min_thr_sc_area
    fd[fa_l_thr_ii] = np.nan
    fl[fa_l_thr_ii] = np.nan
    # Analyze GIS data for not-NaNs
    gis_val_ii  = np.nonzero(~np.isnan(fd))
    gis_val_iix = gis_val_ii[1]
    gis_val_iiy = gis_val_ii[0]
    gis_val_ct  = gis_val_iiy.shape[0]
    
    # Prepare Lookup table for flow direction
    # prepare flow direction look up table for x and y cell number differences
    fd_lu = pd.DataFrame(
                np.array([[ 1, 0], [ 1, 1], [ 0, 1], [-1, 1],
                          [-1, 0], [-1,-1], [ 0,-1], [ 1,-1]]),
                index=[1, 2, 4, 8, 16, 32, 64, 128], columns=[f_dx, f_dy])
    # pre-allocate arrays
    if ncols * nrows <= 32767: np_type = np.int32
    else: np_type = np.int64
    fd_xd = np.empty((gis_val_ct, 1), dtype=np_type)
    fd_yd = np.empty((gis_val_ct, 1), dtype=np_type)
    # iterate flow direction integer values
    for fdir, (dx, dy) in fd_lu.iterrows():
        fd_notnans_ii = fd[~np.isnan(fd)] == fdir
        fd_xd[fd_notnans_ii] = dx
        fd_yd[fd_notnans_ii] = dy
    # calculate downstream cell indices (column and row)
    jtm_down_xd = gis_val_iix + np.int64(fd_xd[:, 0])
    jtm_down_yd = gis_val_iiy + np.int64(fd_yd[:, 0])
    
    # Calculate Flow Accumulation differences for every cell 
    if print_out: print('...calculate the flow accumulation differences...')
    # Calculate flow accumulation threshold using D8: 
    # calculate the flow accumulation differences
    fa_dif = np.zeros((nrows, ncols), dtype=np.float64)
    fa_dif[gis_val_ii] = fa[jtm_down_yd, jtm_down_xd] - fa[gis_val_ii]
    # set all flow accumulation values < threshold to nan
    fa_dif[fa_l_thr_ii] = np.nan
    # get flow accumulation difference values for flow accumulation flow paths
    with np.errstate(invalid='ignore'):
        fa_le_thr_ii = fa >= fa_min_thr_sc_area
    # export as raster
    tc.numpy_to_predef_gtiff_fmt(
               fa_dif, xll, yll, cellsz, path_gdb_out, name_fa_dif,
               ctrl_tif_export=ctrl_tif_export, path_gdb_fmt_in=path_gdb_out,
               name_raster_fmt_in=name_fa_c, path_folder_tif_tmp=path_files_out)
    
    # Prepare temporal indices in array format
    # Create temporal index array with continuous number 
    if print_out: print('...create temporal index array...')
    jtm = np.ones((nrows, ncols), dtype=np_type) * -1
    jtm[gis_val_ii] = range(1, gis_val_ct + 1)
    # Calculate the temporal downstream cell array jtm_down using the FlowDir.
    if print_out: print('...calculate the temporal downstream cell array...')
    jtm_down = np.ones((nrows, ncols), dtype=np_type) * -1
    jtm_down[gis_val_ii] = jtm[jtm_down_yd, jtm_down_xd]
    
    # Find the catchment outlet with the largest Flow Accumulation
    if print_out: print('...find the catchment outlet...')
    of_ii = np.nonzero(np.logical_and(jtm != -1, jtm_down == -1))
    # Mark the outlet cell in jtm_down with a 0
    jtm_down[of_ii] = 0
        
    # Prepare temporal indices in vector format
    # Create upstream matrix coordinate list MC_up
    if print_out: print('...create upstream matrix coordinate list...')
    # get temporal indices of cells
    jt_fa = jtm[fa_le_thr_ii]
    # get flow accumulation and length values for flow accumulation flow paths
    fa_jt = pd.Series(fa[fa_le_thr_ii], index=jt_fa, name=f_fa)
    fl_jt = pd.Series(fl[fa_le_thr_ii], index=jt_fa, name=f_fl)
    # get indices of flow accumulation values >= fa_min_thr_sc_area
    jt_fa_arrcoords = pd.DataFrame(np.argwhere(fa_le_thr_ii), index=jt_fa,
                                   columns=[f_row, f_col])
    # get downstream temporal indices of cells
    jt_down_fa = pd.Series(jtm_down[fa_le_thr_ii], index=jt_fa, name=f_jt_down)
    # preallocate lists
    jt_up_fa = pd.Series(jt_fa.shape[0]*[[]], index=jt_fa, name=f_jt_up)
    # iterate over temporal upstream list
    for jt, jt_down in jt_down_fa.iteritems():
        # find all rows in Jtm_down_FA which do have jt as downstream cell
        verw_ii = jt_down_fa == jt
        if not np.all(~verw_ii):
            # get array indices of rows
            jt_up_fa[jt] = jt_fa[verw_ii]
        else:
            jt_up_fa[jt] = np.array([])
    
    # Find relevant confluence pour point values
    # Initialization of algorithm defining starting cell (largest flow length)
    if print_out: print('...find relevant confluence pour point values...')
    fl_jt_down = copy.deepcopy(fl_jt)
    fl_jt_down.name=f_fl_down
    # pre-define pour point raster matrix
    done          = pd.Series(np.ones(jt_fa.shape[0], dtype=np_type) * -1,
                              index=jt_fa, name='done')
    pp_list       = []
    id_pp         = []
    id_pp_down    = []
    id_pp_jt_down = []
    fa_pp         = []
    fl_pp         = []
    # pre-define iterators
    d_ii  = 1
    ws_id = 1
    # Find cell with largest flow length and its temporal index
    fl_max = np.nanmax(fl_jt_down)
    jt     = jt_fa[fl_jt_down == fl_max][0]
    # debug protocol
    if print_out and debug:
        x = jt_fa_arrcoords.at[jt, f_row]
        y = jt_fa_arrcoords.at[jt, f_col]
        print('   Init.: {0:d}/{1:d}'.format(x, y))
    
    # run loop as long as outflow cell is not reached
    while fl_jt_down.at[jt] != 0:
        # get downstream cell id 
        jt_down = jt_down_fa.at[jt]
        # get reference id at recent cell
        jt_ref = jt
        # as long as cell is not a confluence point:
        while jt_up_fa.at[jt_down].shape[0] <= 1:
            # mark cell as done and go downwards
            done.at[jt] = d_ii
            jt = jt_down_fa.at[jt]
            # get downstream cell id 
            jt_down = jt_down_fa.at[jt]
            if fl_jt_down.at[jt] == 0:
                break
        # mark cell as done and go downwards
        done.at[jt] = d_ii
        # get flow accumulation and length of section
        fa_sect = fa_jt.loc[done == d_ii]
        fl_sect = fl_jt.loc[done == d_ii]
        # get temporal ID
        jt_sect = jt_fa[done == d_ii]
        mc_sect = jt_fa_arrcoords.loc[done == d_ii, :]
        # get min and max flow accumulation of section
        if jt_up_fa.at[jt_ref].shape[0] == 0:
            fa_sect_min = 0
            fl_sect_max = fl_max
        else: 
            fa_sect_min = np.min(fa_sect)
            fl_sect_max = np.max(fl_sect)
        fa_sect_max = np.max(fa_sect)
        fl_sect_min = np.min(fl_sect)
        # correct flow accumulation and length for minimum value
        fa_sect_corr = fa_sect     - fa_sect_min
        # calculate flow accumulation difference of recent section
        fa_sect_dif  = fa_sect_max - fa_sect_min
        # calculate number of necessary sub-watersheds
        div_int = np.round(fa_sect_dif / fa_thr_sc_area, 0)
        nb_pp   = div_int - 1
        # pre-define section sub-watershed ID list
        id_pp_sect = []
        # pre-define sub-watershed ID vector for section
        ws_id_sect = np.zeros(fa_sect.shape) * np.nan
        # if more than one sub-catchments have to be created in section:
        if div_int > 1:
            # define flow accumulation threshold
            fa_rat = fa_sect_dif / div_int
            # calculate ratios
            fa_rat_sect = fa_sect_corr / fa_rat
            # define numbers to search for
            thr_pp = np.arange(1, nb_pp + 1)
            # pre-define section ID list
            jti_pp_sect = []
            # add sub-watershed pour points
            for thr in thr_pp:
                # find index of pour point
                jti_pp_sect.append(np.argmin(np.abs(fa_rat_sect - thr)))
                # add pour point
                mc_pp = mc_sect.iloc[jti_pp_sect[-1], :]
                pp_list.append(mc_pp.to_numpy())
                # calculate true flow accumulation value at PP
                fa_jt_pp = fa_sect.iloc[jti_pp_sect[-1]]
                fa_pp.append(fa_jt_pp)
                # calculate true flow length value at PP
                fl_jt_pp = fl_sect.iloc[jti_pp_sect[-1]]
                fl_pp.append(fl_jt_pp)
                # calculate sub-catchment area and set sub-catchment ID
                if thr == 1:
                    a_jt_pp = fa_jt_pp - fa_sect_min
                    ws_id_sect[:jti_pp_sect[-1]] = ws_id
                else:
                    a_jt_pp = fa_jt_pp - fa_pp[-2]
                    ws_id_sect[jti_pp_sect[-2]:jti_pp_sect[-1]] = ws_id
                # add to protocol
                id_pp.append(ws_id)
                id_pp_sect.append(ws_id)
                # add upstream subcatchment ID to recent ID
                if thr == 1:
                    for pp_jt, pp_id in id_pp_jt_down:
                        if np.any(np.in1d(pp_jt, jt_sect)):
                            if print_out and debug: 
                                print('{0:d} -> {1:d}'.format(pp_id, ws_id))
                            id_pp_down.append([pp_id, ws_id])
                else:
                    if print_out and debug: 
                        print('{0:d} -> {1:d}'.format(ws_id-1, ws_id))
                    id_pp_down.append([ws_id-1, ws_id])
                # increment iterator
                ws_id = ws_id + 1
                # debug protocol
                if print_out and debug:
                    x = mc_pp[0]
                    y = mc_pp[1]
                    print(('   Bran.: {0:5d}/{1:5d}, fa: min: {2:9.0f}, '
                           'max: {3:9.0f}, PP: {4:9.0f}, A: {5:7.0f}m²').format(
                            x, y, fa_sect_min, fa_sect_max, fa_jt_pp, a_jt_pp))
        else:
            if print_out and debug: print('   no Bran.')
        # calculate flow length difference for section
        fl_dif = fl_sect_max - fl_sect_min
        if fl_dif >= fl_min_tol or fl_jt_down.at[jt] == 0:
            # add pour point above confluence point
            pp_list.append(jt_fa_arrcoords.loc[jt, :].to_numpy())
            # calculate true flow accumulation value at PP
            fa_jt_pp = fa_jt.at[jt]
            fa_pp.append(fa_jt_pp)
            # calculate true flow length value at PP
            fl_jt_pp = fl_jt.at[jt]
            fl_pp.append(fl_jt_pp)
            # calculate sub-catchment area and set sub-catchment ID
            if div_int > 1:
                a_jt_pp = fa_jt_pp - fa_pp[-2]
                ws_id_sect[jti_pp_sect[-1]:] = ws_id
            else:
                a_jt_pp = fa_jt_pp - fa_sect_min
                ws_id_sect[:] = ws_id
            # add to protocol
            id_pp.append(ws_id)
            # pre-announce inflow subcatchment ID to downstream cell
            id_pp_jt_down.append([jt_down, ws_id])
            # add upstream subcatchment ID to recent ID
            if div_int > 1:
                if print_out and debug:
                    print('{0:d} -> {1:d}'.format(ws_id - 1, ws_id))
                id_pp_down.append([ws_id - 1, ws_id])
            else:
                for pp_jt, pp_id in id_pp_jt_down:
                    if np.any(np.in1d(pp_jt, jt_sect)):
                        if print_out and debug:
                            print('{0:d} -> {1:d}'.format(pp_id, ws_id))
                        id_pp_down.append([pp_id, ws_id])
            # increment iterator
            ws_id = ws_id + 1
            # debug protocol
            if print_out and debug:
                x = jt_fa_arrcoords.at[jt, 'row']
                y = jt_fa_arrcoords.at[jt, 'col']
                print(('   Conf.: {0:5d}/{1:5d}, fa: min: {2:9.0f}, '
                       'max: {3:9.0f}, PP: {4:9.0f}, A: {5:7.0f}m²').format(
                        x, y, fa_sect_min, fa_sect_max, fa_jt_pp, a_jt_pp))
        else:
            # set sub-catchment ID to 0 as inidcator of missing connection
            ws_id_sect[:] = 0
            # pre-announce inflow subcatchment IDs to downstream cell
            for pp_jt, pp_id in id_pp_jt_down:
                if np.any(np.in1d(pp_jt, jt_sect)):
                    if print_out and debug:
                        print('{0:d} -> #'.format(pp_id))
                    id_pp_jt_down.append([jt_down, pp_id])
            if print_out and debug:
                print('   no Conf.')
        # exclude branch from flow length search
        fl_jt_down[done == d_ii] = np.nan
        if np.all(np.isnan(fl_jt_down)):
            break
        # Find cell with largest flow length and its temporal index
        jt = jt_fa[fl_jt_down == np.nanmax(fl_jt_down)][0]
        # increment iterator
        d_ii += 1
    # set outflow downstream PP
    id_pp_down.append([pp_id, 0])
    
    # Create DataFrame
    # create Series of ID number
    pp_ser = pd.Series(id_pp, index=id_pp, name=f_pp, dtype=np.int64)
    # sort temporary downstream ID list and convert to numpy array
    pp_down_arr = np.asarray(id_pp_down, dtype=np.int64)
    pp_down_ser = pd.Series(pp_down_arr[:, 1], index=pp_down_arr[:, 0],
                               name=f_pp_down, dtype=np.int64).sort_index()
    # calculate coordinates
    pp_coords_df = pd.DataFrame(pp_list, index=id_pp, columns=[f_y, f_x],
                                dtype=np.float64)
    pp_coords_df.x = xll + (        pp_coords_df.x + 0.5) * cellsz
    pp_coords_df.y = yll + (nrows - pp_coords_df.y - 0.5) * cellsz
    # create DataFrame
    pp_points_df = pd.concat([pp_coords_df, pp_ser, pp_down_ser, 
                              pd.DataFrame(np.stack([fa_pp, fl_pp], axis=1),
                                           index=id_pp, columns=[f_fa, f_fl])],
                              axis=1).astype({
                                      f_x : np.float64, f_y      : np.float64,
                                      f_pp: np.int64  , f_pp_down: np.int64  ,
                                      f_fa: np.int64  , f_fl     : np.float64})
    
    # export DataFrame to point feature class
    # get spatial reference object
    sr_obj = arcpy.Describe(path_dem).spatialReference
    # export to point feature classes
    tc.tgb_to_points(pp_points_df, sr_obj, path_gdb_out, name_pp_sc,
                     geometry_fields=(f_x, f_y))
    
    # Calculate subcatchments and flow accumulation network
    if print_out: print('...calculate subcatchments...')
    # calculate pour point watershed polygons
    # set environments
    path_fa_c = path_gdb_out + name_fa_c
    arcpy.env.extent = path_fa_c
    arcpy.env.snapRaster = path_fa_c
    # snap pour points to flow accumulation
    if arcpy.Exists(path_ssp_sc): arcpy.Delete_management(path_ssp_sc)
    path_pp_sc = path_gdb_out + name_pp_sc
    spp_sc = arcpy.sa.SnapPourPoint(path_pp_sc, path_fa_c, 0, f_pp)
    spp_sc.save(path_ssp_sc)
    spp_sc = None
    # calculate raster watershed representation
    if arcpy.Exists(path_sc_ws_r): arcpy.Delete_management(path_sc_ws_r)
    sc_ws_r = arcpy.sa.Watershed(name_fd_c, path_ssp_sc, f_val)
    sc_ws_r.save(path_sc_ws_r)
    sc_ws_r = None
    # calculate polygon watershed representation
    path_sc_ws_s = path_gdb_out + name_sc_ws_s
    if arcpy.Exists(path_sc_ws_s): arcpy.Delete_management(path_sc_ws_s)
    arcpy.conversion.RasterToPolygon(path_sc_ws_r, path_sc_ws_s, 'NO_SIMPLIFY',
                                     f_val, 'SINGLE_OUTER_PART', None)
    
    # calculate flow network from flow accumulation using watershed threshold
    if print_out: print('...calculate flow network using watershed threshold...')
    fnw_fa_r = arcpy.ia.Con(name_fa_c, 1, 0,
                            '{0:s} >= {1:.0f}'.format(f_val, fa_min_thr_sc_area))
    fnw_fa_r.save(path_fnw_fa_r)
    fnw_fa_r = None
    arcpy.conversion.RasterToPolyline(path_fnw_fa_r, name_fnw_fa, 'ZERO', 0,
                                      'NO_SIMPLIFY', f_val)
    
    return pp_points_df

# %% calculate parameters for export of the tgb.dat file
def calc_params(hq_ch, hq_ch_a,
                path_dem_c, path_fd_c, path_fa_c, path_fl_c, path_pp_sc, 
                path_files_out, path_gdb_out, 
                name_tgb_p='tgb_p', name_tgb_sj='tgb_sj', 
                def_sc_area=1000000, def_a_min_tol=50,
                def_zmin_rout_fac=0.5, def_sl_excl_quant=0.999, 
                ch_est_method='combined', ser_q_in_corr=None, 
                def_bx=0, def_bbx_fac=1, def_bnm=1.5,
                def_bnx=100, def_bnvrx=4, def_skm=30, def_skx=20, 
                ctrl_rout_cs_fit=False,
                def_cs_dist_eval=50, 
                def_cs_wmax_eval=600, def_cs_hmax_eval=10,
                def_flpl_wmax_eval=50,
                def_ch_wmax_eval=40, def_ch_hmin_eval=0.1, def_ch_hmax_eval=3.0,
                def_ch_hmin=0.2, def_ch_vmin=0.5, def_ch_vmax=3.0,
                def_chbank_slmin=0.1,
                def_val_wres=10.0, def_flpl_wres=0.20,
                def_ch_w=0.5, def_ch_h=0.5, def_ch_wres=0.05,
                def_lam_hres=0.1, 
                ctrl_show_plots=False, ctrl_save_plots=False, path_plots_out=None,
                print_out=False):
    """
    This function calculates all necessary spatial parameters for runoff
    concentration and routing modules in LARSIM, which are stored later in the 
    tgb.dat file. 
    In the standard version, the river cross section is parametrized using 
    channel estimator functions. Alternatively, it delineates, averages and
    fits multiple cross sections from a high-resolution digital elevation grid
    to achieve a realistic channel, river bank and foreland representation.

    JM 2021

    Arguments:
    -----------
    hq_ch: float [m³/s]
        HQ2 (or MHQ summer) of an existing gauge close to the modeled catchment
    hq_ch_a: float [km²]
        cathcment area of an existing gauge close to the modeled catchment
    path_dem_c: str
        path of the elevation raster within model domain
    path_fd_c: str
        path of the flow direction raster within model domain
    path_fa_c: str
        path of the flow accumulation raster within model domain
    path_fl_c: str
        path of the flow length raster within model domain
    path_pp_sc: str
        path of the subcatchment pour points' feature class
    path_files_out: str
        storage path for intermediate data (e.g., 'c:\tmp_model_data\')
    path_gdb_out: str
        path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
    name_tgb_p: str (optional, default: 'tgb_p')
        file name, where the feature class shall be stored
    name_tgb_sj: str (optional, default: 'tgb_sj')
        file name, where the feature class shall be stored
    def_sc_area: integer (optional, default: 10**6)
        intended size of the resulting optimized subwatershed model elements
        attention: From its construction, the algorithm tends to underestimate
        the element size and gives in the mean 15% smaller model elements due to
        tendencially very small elements at confluence points [m²]
    def_a_min_tol: float (optional, default: 50)
        minimum subcatchment area tolerated [m²]
    def_zmin_rout_fac: float  (optional, default: 0.5)
        define lower elevation for routing elements as percentage of stream's [-]
        low and high point
    def_sl_excl_quant: float (optional, default: 0.999)
        upper threshold for realistic slope as quantile
    ch_est_method: string (optional, default: 'combined')
        String defining channel estimation function. Possible values: 
        - 'Allen': Allen et al. (1994)
        - 'Krauter': Krauter (2006)
        - 'combined': Allen et al.(1994) for small and Krauter (2006) for large areas
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
    def_skm: float (optional, default: 30 = natural river channel, vegetated river bank)
        Float defining the Strickler roughness values in the channel [m1/3s-1]
    def_skx: float (optional, default: 20 = uneven vegetated foreland)
        Float defining the Strickler roughness values of the left and right
        foreland [m1/3s-1]
    ctrl_rout_cs_fit: boolean (optional, default: False)
        (de-)activate fitting of triple trapezoid cross section profiles
    def_cs_dist_eval: int (optional, default: 50) [m]
        Linear distance between two automatically derived, consecutive 
        cross sections. This value should be oriented at the desired size
        of subcatchments to generate at least few cross sections.
    def_cs_wmax_eval: int (optional, default: 600) [m]
        Length of the automatically generated cross sections perpendicular
        to the flow accumulation flow network. It should cover the valley
        at least until the estimated maximum water depth.
    def_cs_hmax_eval: float (optional, default: 10) [m]
        maximum height of cross section evaluation
    def_flpl_wmax_eval: float (optional, default: 50) [m]
        estimated maximum flood plain width
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
    def_val_wres: float (optional, default: 10 [m])
        resolution of interpolated points within vally
    def_flpl_wres: float (optional, default: 0.20 [m])
        resolution of interpolated points within flood plain
    def_ch_w, def_ch_h: float (optional, default: 0.5, 0.5) [m]
        artificial channel width (w) and depth (h), added to continuiously
        descending cross sections
    def_ch_wres: float (optional, default: 0.05 [m])
        resolution of interpolated points within channel
    def_lam_hres: float (optional, default: 0.1) [m]
        spacing between evaluation lamellae
    ctrl_show_plots: boolean (optional, default: False) [-]
        (de-)activate pop-up of figures
    ctrl_save_plots: boolean (optional, default: False) [-]
        (de-)activate export of figures as files
    path_plots_out: str (optional, default: None)
        path where plots are stored (e.g., 'c:\model_creation\fig')    
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
    """
    # %% establish downstream relation of pour points
    def get_downstream_pp(ser_pp_fl, ser_pp_row, ser_pp_col,
                          arr_fd, gis_val_rows, gis_val_cols,
                          gis_val_down_rows, gis_val_down_cols,
                          print_out=False):
        """
        This function finds all downstrem subcatchment pour points using the 
        flow direction and accumulation moving downstream from one to the other.
    
        JM 2021
    
        Arguments:
        -----------
        ser_pp_fl: pandas.Series
            Series of flow length values corresponding to the serie's
            ascending pour point index. The outflow pour point is flow length = 0
            (e.g., pd.Series([76.5, 40.3, 43.2, 0],
                             index=[1, 2, 3, 4], name='pp_fl'))
        ser_pp_row, ser_pp_col: pandas.Series
            Series of row and column indices, which represent the pour points'
            position in GIS data arrays (e.g., flow accumulation)
            (e.g., pd.Series([0, 0, 1, 2], index=[1, 2, 3, 4], name='row' or 'col'))
        arr_fd: numpy.array
            Array of flow direction covering all pour points.
            (e.g., np.array([[2, 4], [4, 32]]))
        gis_val_rows, gis_val_cols: numpy.array (vectors)
            Vectors of row and column indices, which represent cells in the flow
            accumulaion array, which are above the flow accumulation threshold
            (e.g., np.array([[0, 1], [1, 1]]))
        gis_val_down_rows, gis_val_down_cols: numpy.array (vectors)
            Vectors of row and column indices, which represent the downstream
            cells of all cells in the flow accumulaion array, which are above
            the defined flow accumulation threshold
            (e.g., np.array([[1, 1], [1, 2]]))
        print_out: boolean (optional, default: False)
            true if workprogress shall be print to command line
    
        Returns:
        -----------
        ser_pp_down: pandas.Series
            Series of downstream pour point IDs, outflow is marked with a zero
            (e.g., pd.Series([2, 4, 4, 0], index=[1, 2, 3, 4], name='pp_down'))
        """
        # define internal variable fields
        f_pp_down = 'pp_down'
        # (de-)activate debug output in command line
        debug = False
        # pre-allocate arrays
        pp_sc_id = ser_pp_fl.index.values
        ser_pp_down = pd.Series(np.zeros((pp_sc_id.shape[0],)),
                                index=ser_pp_fl.index,
                                name=f_pp_down).astype(np.int)
        # initialize loop with pour point with largest flow length
        pp = ser_pp_fl.idxmax()
        # remove flow length value from pour point list
        ser_pp_fl.loc[pp] = np.nan
        # initialize counter of done pour points
        pp_ct = 1
        # loop through pour points
        while pp_ct <= np.max(pp_sc_id):
            # get pour point array row and column
            row = ser_pp_row.at[pp]
            col = ser_pp_col.at[pp]
            # get flow direction
            pp_fd = arr_fd[row, col]
            # go downstream along flow direction as long as no other
            # pour point is met
            while not np.isnan(pp_fd):
                # find index of cell in non-nan list
                cell_ii = np.nonzero(np.all((
                        gis_val_rows == row,
                        gis_val_cols == col), 0))
                if cell_ii[0].shape[0] == 0:
                    break
                # get downstream row and column
                row = gis_val_down_rows[cell_ii][0]
                col = gis_val_down_cols[cell_ii][0]
                # test, if at the current location is a pour point
                pp_test_bool = np.all((
                        np.isin(ser_pp_row, row), 
                        np.isin(ser_pp_col, col)), 0)
                if np.any(pp_test_bool):
                    break
            if pp_ct < np.max(pp_sc_id):
                # get downstream pour point ID
                pp_down_id = pp_sc_id[np.all((
                        ser_pp_row == row,
                        ser_pp_col == col), 0)][0]
                ser_pp_down.at[pp] = pp_down_id
                if debug and print_out:
                    print(pp, ' -> ', pp_down_id)
                # find new pour point with largest flow length
                pp = ser_pp_fl.idxmax()
                # remove flow length value from pour point list
                ser_pp_fl.loc[pp] = np.nan
            # increment counter
            pp_ct += 1
        if debug and print_out:
            print(pp, ' -> ', 'outflow')
        return ser_pp_down
    
    # %% establish upstream relation of pour points
    def get_upstream_pp(ser_pp_down):
        """
        This function finds all upstream model elements using the index and the
        downstream relation.
    
        JM 2021
    
        Arguments:
        -----------
        ser_pp_down: pandas.Series
            Series of downstream pour point indices corresponding to the serie's
            ascending index. The last value is outlet, identified with a zero.
            The outlet will be neglected in calculations.
            (e.g., pd.Series([2, 4, 4, 0], index=[1, 2, 3, 4], name='pp_down'))
    
        Returns:
        -----------
        df_pp_up: pandas.DataFrame
            DataFrame of corresponding upstream pour point indices. The number
            of columns depends on the maximum number of upstream pour points.
            (e.g., pd.DataFrame([[-1, -1], [1, -1], [-1, -1], [2, 3]],
                                index=[1, 2, 3, 4], columns=[1, 2]))
        """
        # pre-allocate arrays
        max_up_nb = 2
        df_pp_up = pd.DataFrame(
                np.zeros((ser_pp_down.index.shape[0], max_up_nb)) - 1,
                index=ser_pp_down.index, 
                columns=range(1, max_up_nb + 1)).astype(np.int)
        # iterate pour points
        for pp, pp_down in ser_pp_down.iteritems():
            if pp_down != 0:
                # get upstream list
                pp_up = df_pp_up.loc[pp_down, :]
                pp_up_open_bool = pp_up == -1
                # extend DataFrame if neccessary
                if not np.any(pp_up_open_bool):
                    df_pp_up[np.max(df_pp_up.columns) + 1] = pd.Series(
                            np.zeros((df_pp_up.index.shape[0], )) - 1,
                            index=df_pp_up.index).astype(np.int)
                    pp_up = df_pp_up.loc[pp_down, :]
                    pp_up_open_bool = pp_up == -1
                # add index to upstream list at first open position (=-1)
                pp_up.at[pp_up_open_bool.idxmax()] = pp
                df_pp_up.loc[pp_down, :] = pp_up
        return df_pp_up
    
    # %% calculate final model network
    def calc_model_nw(df_pp_sc, df_pp_up,
                      str_headw, str_routing, str_dummy, val_no_dummy,
                      print_out=False):
        """
        This function converts all pour point indices to model element indices
        respecting LARSIM conventions and returns two DataFrames. 
        These contain all necessary variables representing the model structure.
    
        JM 2021
    
        Arguments:
        -----------
        df_pp_sc: pandas.DataFrame
            DataFrame of of pour point indices including information about:
                - pp_down: downstream pour point indices
                    (e.g., [2, 4, 4, 0])
                - fa: pour points' flow accumulation values
                    (e.g., [456, 1323, 1456, 2509])
                - fl: pour points' flow length values
                    (e.g., [4563, 3422, 3214, 0])
            (e.g., pd.DataFrame((see individual variables),
                                index=[1, 2, 3, 4], columns=['pp_down', 'fa', 'fl']))
        df_pp_up: pandas.DataFrame
            DataFrame of corresponding upstream pour point indices. The number
            of columns depends on the maximum number of upstream pour points.
            (e.g., pd.DataFrame([[-1, -1], [1, -1], [-1, -1], [2, 3]],
                                index=[1, 2, 3, 4], columns=[1, 2]))
        str_headw, str_routing, str_dummy: str
            strings representing the model element types 'headwater', 'routing'
            and 'dummy' element. (e.g., 'headwater', 'routing', 'dummy')
        val_no_dummy: int
            default value for no-dummy elements in variable 'tgb_dtgb' of 
            the tgb_p point feature class [-] (e.g., -1)
        print_out: boolean (optional, default: False)
            true if workprogress shall be print to command line
    
        Returns:
        -----------
        df_tgb: pandas.DataFrame
            DataFrame of of model element indices including information about:
               - tgb_down: downstream model element indices
                   (e.g., [2, 4, 4, 0])
               - tgb_type: type of model elements
                   (e.g., [headwater, routing, headwater, routing])
               - tgb_tgbt: corresponding pour point indices
                   (e.g., [1, 2, 3, 4])
               - tgb_dtgb: corresponding real element indices for dummies
                   (e.g., [-1, -1, -1, -1])
               - nrflv: name of model element consisting of type character
                   and element ID
        df_tgb_up: pandas.DataFrame
            DataFrame of corresponding upstream model element indices. The number
            of columns depends on the maximum number of upstream pour points.
            (e.g., pd.DataFrame([[-1, -1], [1, -1], [-1, -1], [2, 3]],
                                index=[1, 2, 3, 4], columns=[up1, up2]))
        """
        # define internal field names
        f_pp_down  = 'pp_down'
        f_fa       = 'fa'
        f_fl       = 'fl'
        f_tgb_down = 'tgb_down'
        f_tgb_type = 'tgb_type'
        f_tgb_dtgb = 'tgb_dtgb'
        f_tgb_tgbt = 'tgb_tgbt'
        f_nrflv    = 'nrflv'
        f_tgb_up1  = 'up1'
        f_tgb_up2  = 'up2'
        # define key-words to identify element types
        str_headw   = 'headwater'
        str_routing = 'routing'
        str_dummy   = 'dummy'
        # create real representative index list for dummy subcatchments
        def real_repr_idx(df_tgb, str_dummy, print_out=False):
            
            if print_out: print(('...create representative index list for '
                                 'dummy subcatchments...'))
            # Preallocate arrays
            ser_tgb_dtgb = pd.Series(np.ones(df_tgb.shape[0]) * val_no_dummy,
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
        
        # iterating from largest flow length downstream to outlet
        # (tree-climbing algorithm)
        if print_out: print('...calculate final flow network...')
        
        debug = False
        # define inputs
        Jt      = np.int64(df_pp_sc.index.values)
        Jt_up   = np.int64(np.hstack((np.array(df_pp_up),
                                      np.zeros((df_pp_up.shape[0],
                                                7 - df_pp_up.shape[1])) - 1)))
        Jt_down = np.array(df_pp_sc.loc[:, f_pp_down])
        Jt_FA   = np.array(df_pp_sc.loc[:, f_fa])
        Jt_FL   = np.array(df_pp_sc.loc[:, f_fl])
        # calculate sum of necessary dummy cells (which have >2 upstream cells)
        D_count = np.nansum(Jt_up[:, 2:7] != -1)
        # calculate number of temporal index numbers jt
        Jt_count= Jt_up.shape[0]
        # calculate number of final indices j as sum of dummy and real cells
        J_count = Jt_count + D_count
        # preallocate lists for final indices J, J_type and J_jt, final upstream 
        # and downstream lists J_up and J_down, and protocol list Done
        J_type  = J_count * [None]
        J       = np.array(range(1, J_count+1))
        J_up    = np.ones((J_count, 2),     dtype=np_type) * -1
        J_down  = np.ones((J_count,  ),     dtype=np_type) * -1
        J_jt    = np.ones((J_count, 1),     dtype=np_type) * -1
        Done    = np.ones((np.nanmax(Jt)),  dtype=np_type) * -1
        # calculate protocol list D_contr
        D_contr = np.nansum(Jt_up[:,2:] != -1, 1)
        
        # initialze loop
        if print_out: print('...initialize...')
        # Find cell with largest flow length and its temporal index
        jti = np.argmax(Jt_FL)
        jt  = Jt[jti]
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
            print('   Initial subcatchment:{0:4d}'.format(jt))
            
        # while either outlet is not reached or not all upstream members
        # are processed
        while Jt_down[jti] != 0 or ssnotdone.shape[0] != 0:
            # case 1: HEADWATER CELL as ssnotnan is empty
            # -> create new index for headwater subcatchment and move downwards
            if ss.shape[0] == 0:
                # increment final index, fill type and link lists
                jj          += 1
                jji          = jj - 1
                J_type[jji]  = str_headw
                J_jt[jji, 0] = jt
                # debug protocol
                if debug and print_out:
                    print('tgb:{0:4d},{1:4d} ->{2:4d}, {3:s} subcatchment'.format(
                            jj, jt, Jt_down[jti], str_headw))
                # mark cell as done and go downwards
                Done[jti] = 1
                jt        = Jt_down[jti]
                jti       = np.nonzero(Jt == jt)[0][0]
                # debug protocol
                if debug and print_out:
                    print('   -> down to {0:4d}'.format(jt))
            else:
                # case 2: ROUTING CELL as all upstream cells are done
                # -> create new index for routing subcatchment and move downwards
                if all(Done[ssi] == 1):
                    # increment final index, fill type and link lists
                    jj += 1
                    jji = jj - 1
                    J_type[jji] = str_routing
                    J_jt[jji,0] = jt
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
                        pr_up = Jt_up[jti,:]
                        pr_up = pr_up[pr_up != -1]
                        print(('tgb:{0:4d},{1:4d} ->{2:4d}, '
                               'Jt_up: {3:s}, {4:s} subc.').format(
                                       jj, jt, Jt_down[jt-1],
                                       str(pr_up[~np.isnan(pr_up)])[1:-1],
                                       str_routing))
                    # mark cell as done and go downwards
                    Done[jti] = 1
                    jt        = Jt_down[jti]
                    jti       = np.nonzero(Jt == jt)[0][0]
                    # debug protocol
                    if debug and print_out:
                        print('   -> down to {0:4d}'.format(jt))
                else:
                    # case 3: DUMMY CELL as not all required dummy cells are
                    # done but >= 2 upstream cells are done
                    # -> create new index for dummy cell and move upwards to
                    # the cell with the largest flow accumulation
                    if np.sum(Done[ssi] != -1) >= 2:
                        # increment final index, fill type and link lists
                        jj          += 1
                        jji          = jj - 1
                        J_type[jji]  = str_dummy
                        J_jt[jji, 0] = 0
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
                            print(('tgb:{0:4d},{1:4d} ->{2:4d}, '
                                   'Jt_up: {3:s}, {4:s} subc.').format(
                                           jj, jt, Jt_down[jt - 1],
                                           str(pr_up[~np.isnan(pr_up)])[1:-1],
                                           str_dummy))
                        # decrement dummy protocol variable
                        D_contr[jti] = D_contr[jti] - 1
                    # case 4 (else): UPWARDS MOVEMENT as not all required dummy
                    # cells are done and < 2 upstream cells are done
                    # -> do not create new index
                    
                    # calculate not done subset of upstream cells and its
                    # largest flow acc. cell
                    # preallocate subset for flow accumulation calculation
                    ssflowacc = np.zeros((ssnotdone.shape[0]), dtype=np_type)
                    # iterate not done subset of upstream cells and find
                    # flow accumulation
                    for ii,iiv in enumerate(ssflowacc):
                        ssflowacc[ii] = Jt_FA[Jt == ssnotdone[ii]]
                    # calculate temporal index of max. flow accumulation 
                    ssmaxind = ssnotdone[ssflowacc == np.amax(ssflowacc)]
                    # go upstream to max flow acc or first cell if more than
                    # one solutions exist
                    jt  = ssmaxind[0]
                    jti = np.nonzero(Jt == jt)[0][0]
                    # debug protocol
                    if debug and print_out:
                        print('   -> up to {0:4d}'.format(jt))
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
        if len(J) > 1:
            J_type[jji] = str_routing
        else:
            J_type[jji] = str_headw
        # debug protocol
        if debug and print_out:
            pr_up = Jt_up[jti, :]
            pr_up = pr_up[pr_up != -1]
            if len(J) > 1:
                print('tgb:{0:4d},{1:4d} ->{2:4d}, Jt_up: {3:s}, {4:s} subc.'.format(
                    jj, jt, Jt_down[jt - 1], str(pr_up[~np.isnan(pr_up)])[1:-1],
                    str_routing))
            else:
                print('tgb:{0:4d},{1:4d} ->{2:4d}, Jt_up: none, {3:s} subc.'.format(
                    jj, jt, Jt_down[jt - 1], str_headw))
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
        
        # create pandas data frames
        structarr_tgb_in = list(zip(J_down, J_type, np.squeeze(J_jt)))
        df_mn = pd.DataFrame(structarr_tgb_in, index=J,
                             columns=[f_tgb_down, f_tgb_type, f_tgb_tgbt])
        df_tgb_up = pd.DataFrame(J_up, index=J, columns=[f_tgb_up1, f_tgb_up2])
        # create real representative index list for dummy subcatchments
        ser_tgb_dtgb = real_repr_idx(df_mn, str_dummy, print_out=print_out)
        # create names of model subcatchments
        ser_nrflv = pd.Series(df_mn.shape[0]*'', index=df_mn.index, name=f_nrflv)
        for tgb, el_type in df_mn.loc[:, f_tgb_type].iteritems():
            ser_nrflv.at[tgb] = el_type[0].upper() + '{0:05d}'.format(tgb)
            
        df_tgb = pd.concat([df_mn, ser_tgb_dtgb, ser_nrflv], axis=1)
        return df_tgb, df_tgb_up
    
    # %% calculate geo-information for channel routing
    def calc_ch_gis(ser_tgb_up_nd, ser_tgb_type,
                    ser_fa, ser_fl, ser_hu,
                    str_headw, str_routing):
        """
        This function pre-processes the geo-information for channel routing. 
        
        JM 2021
    
        Arguments:
        -----------
        ser_tgb_up_nd: pandas.Series
            Series of corresponding upstream model element indices ignoring dummy
            elements. These are represented as empty array (e.g., []).
        ser_tgb_type: pandas.Series
            Series of model elements' types corresponding to the ascending index. 
            (e.g., pd.Series(['headwater', 'routing', 'headwater', 'routing'],
                             index=[1, 2, 3, 4], name='tgb_type'))
        ser_fa: pandas.Series [-]
            Series of model elements' flow accumulation values.
            (e.g., pd.Series(np.array([456, 1323, 1456, 2509]),
                             index=[1, 2, 3, 4], name='fa'))
        ser_fl: pandas.Series [m]
            Series of model elements' flow length values.
            (e.g., pd.Series(np.array([4563, 3422, 3214, 0]),
                             index=[1, 2, 3, 4], name='fl'))
        ser_hu: pandas.Series [m]
            Series of model elements' minimum elevation values.
            (e.g., pd.Series(np.array([45.3, 23.4, 25.6, 12.2]),
                             index=[1, 2, 3, 4], name='hu'))
        str_headw, str_routing, str_dummy: str
            Strings representing the model elements' types in ser_tgb_type
            (e.g., 'headwater', 'routing', 'dummy')
    
        Returns:
        -----------
        df_tgb_ch_gis: pandas.DataFrame
            DataFrame of element channels' geo-information variables:
               - a: model element area
               - l: model element channel length
               - ch_zmin: minimum channel elevation
               - ch_zmax: maximum channel elevation
        """
        # define internal field names
        f_a      = 'a'
        f_l      = 'l'
        f_chzmin = 'ch_z_min'
        f_chzmax = 'ch_z_max'
        # create DataFrame to store information
        df_tgb_ch_gis = pd.DataFrame(np.zeros((ser_tgb_type.shape[0], 4)) * np.nan, 
                                          index=ser_tgb_type.index,
                                          columns=[f_a, f_l, f_chzmin, f_chzmax])
        # print minimal channel elevation
        df_tgb_ch_gis.loc[:, f_chzmin] = ser_hu
        # summarize inputs for iteration
        df_tgb_in = pd.concat([ser_tgb_type, ser_fa, ser_fl], axis=1)
        # iterate model subcatchments
        for tgb, (tgb_type, tgb_fa, tgb_fl) in df_tgb_in.iterrows():
            # headwater subc.: calculate area, set length to 1 [m] (to ignore routing)
            if tgb_type == str_headw:
                df_tgb_ch_gis.at[tgb, f_a] = tgb_fa
                df_tgb_ch_gis.at[tgb, f_l] = 1
            # routing subc.: calculate information from upstream catchments
            elif tgb_type == str_routing:
                # get upstream subcatchment IDs
                tgb_up = ser_tgb_up_nd.at[tgb]
                # get flow accumulation and flow length of upstream cell
                fa_up = ser_fa.loc[tgb_up]
                fl_up = ser_fl.loc[tgb_up]
                # get height of upstream cell
                hu_up = ser_hu.loc[tgb_up]
                # calculate subcatchment area and flow length difference
                df_tgb_ch_gis.at[tgb, f_a     ] = tgb_fa - np.sum(fa_up)
                df_tgb_ch_gis.at[tgb, f_l     ] = np.average(fl_up, None, fa_up) - tgb_fl
                df_tgb_ch_gis.at[tgb, f_chzmax] = np.average(hu_up, None, fa_up)
            # dummy subc.: set area to 0 [km²] and length to 1 [m] (to ignore routing)
            else:
                df_tgb_ch_gis.at[tgb, f_a] = 0
                df_tgb_ch_gis.at[tgb, f_l] = 1
                
        return df_tgb_ch_gis
    
    # %% Calculate final routing flow lengths for model elements
    def calc_ch_rout_fl(ser_tgb_type, ser_tgb_down, ser_ch_l,
                        str_headw, str_routing, str_dummy):
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
        ser_ch_l: pandas.Series [m]
            DataFrame of corresponding channel flow length values. The Series
            includes the model element ID as index:
            (e.g., pd.Series(data=[344.2, 236.1, 245.2, 0], index=[1, 2, 3, 4],
            name='fl'))
        ser_tgb_down: pandas.Series
            Series of corresponding downstream model element indices.
            (e.g., pd.Series(data=[2, 4, 4, 0], index=[1, 2, 3, 4], name='tgb_down'))
        ser_tgb_type: pandas.Series
            Boolean Series, which identifies the headwater cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[1, 1, 0, 0], index=[1, 2, 3, 4], name='tgb_type'))
        str_headw, str_routing, str_dummy: str
            Strings representing the model elements' types in ser_tgb_type
            (e.g., 'headwater', 'routing', 'dummy')
    
        Returns:
        -----------
        df_tgb_rout_fl: pandas.DataFrame
            DataFrame of corresponding runoff concentration parameters. The DataFrame
            includes the model element ID as index and the following columns:
                - corresponding lower cumulative flow length values (kmu) [m]
                - corresponding upper cumulative flow length values (kmo) [m]
        """
        # define internal field names
        f_lower = 'lower'
        f_mean  = 'mean'
        f_upper = 'upper'
        f_kmu_d = 'kmu_dummy'
        f_kmo_d = 'kmo_dummy'
        f_kmo_h = 'kmo_headw'
        f_kmu   = 'kmu'
        f_kmo   = 'kmo'
        # Calculate real channel flow length values
        # preallocate pandas DataFrame
        df_fl = pd.DataFrame(np.zeros((df_tgb.shape[0], 3)), index=df_tgb.index,
                             columns=[f_lower, f_mean, f_upper])
        # iterate all cells
        for tgb in reversed(ser_tgb_down.index.values):
            # DUMMY CELLS
            if ser_tgb_type.at[tgb] == str_dummy:
                # take value of downstream cell
                tgb_down = ser_tgb_down.at[tgb]
                # take KMO from downstream cell
                df_fl.at[tgb, f_lower] = df_fl.at[tgb_down, f_upper]
            # ROUTING AND HEADWATER CELLS
            else:
                # find real downstream cell
                tgb_down_nd = ser_tgb_down.at[tgb]
                # if recent element is not the outlet element
                if tgb != np.amax(ser_tgb_down.index):
                    # take KMO from downstream cell
                    df_fl.at[tgb, f_lower] = df_fl.at[tgb_down_nd, f_upper]
                # HEADWATER CELLS
                if ser_tgb_type.at[tgb] == str_headw:
                    # use downstream KMO (no routing in upstream cells as mentioned)
                    df_fl.at[tgb, f_upper] = df_fl.at[tgb, f_lower]
                # ROUTING CELLS
                elif ser_tgb_type.at[tgb] == str_routing:
                    # calculate sum of downstream KMO and flow length of recent cell
                    df_fl.at[tgb, f_upper] = df_fl.at[tgb, f_lower] + ser_ch_l.at[tgb]
        # calculate mean flow length in subcatchments' routing sections
        df_fl.loc[ser_tgb_type == str_routing, f_mean] = \
            np.mean(df_fl.loc[ser_tgb_type == str_routing, [f_lower, f_upper]], axis=1)
            
        # Calculate adds for dummy and headwater model subcatchment elements
        # preallocate pandas DataFrame
        df_fl_add = pd.DataFrame(
                np.zeros((ser_tgb_down.shape[0], 3)), index=ser_tgb_down.index,
                columns=[f_kmu_d, f_kmo_d, f_kmo_h]).astype(np.int)
        # set outflow flow length value to 1 [m]
        df_fl_add.at[np.amax(ser_tgb_down.index), [f_kmo_d, f_kmu_d]] = 1
        # iterate all cells
        for tgb in reversed(ser_tgb_down.index.values):
            # get upstream cell IDs
            tgb_up = df_tgb_up.loc[tgb, :].values
            tgb_up = tgb_up[tgb_up != -1]
            # if upstream cells exist
            if tgb_up.shape[0] > 0:
                # KMU add of upstream cell is KMO add of recent cell
                df_fl_add.loc[tgb_up, f_kmu_d] = df_fl_add.loc[tgb, f_kmo_d]
                # get logical indices of upstream dummy elements
                tgb_up_dummy   = tgb_up[ser_tgb_type.loc[tgb_up] == str_dummy]
                tgb_up_nodummy = tgb_up[ser_tgb_type.loc[tgb_up] != str_dummy]
                # if upstream cell is a dummy cell, KMO add of upstream dummy
                # is KMU add of recent cell + 1
                df_fl_add.loc[tgb_up_dummy, f_kmo_d] \
                    = df_fl_add.loc[tgb_up_dummy, f_kmu_d] + 1
                # if upstream cell is not a dummy cell, KMO add of upstream cell
                # is KMU add of recent cell
                df_fl_add.loc[tgb_up_nodummy, f_kmo_d] \
                    = df_fl_add.loc[tgb_up_nodummy, f_kmu_d]
        # calculate adds for headwater subcatchments
        df_fl_add.loc[ser_tgb_type == str_headw, f_kmo_h] = 1
        
        # Calculate final routing flow lengths for model elements
        df_tgb_rout_fl = pd.DataFrame(
                np.zeros((ser_tgb_down.shape[0], 2)), index=ser_tgb_down.index,
                columns=[f_kmu, f_kmo]).astype(np.int)
        # Add Dummy and Head Water Adds
        df_tgb_rout_fl.loc[:, f_kmu] = np.int64(np.round(df_fl.lower, 0) 
                          + df_fl_add.kmu_dummy)
        df_tgb_rout_fl.loc[:, f_kmo] = np.int64(np.round(df_fl.upper, 0) \
                          + df_fl_add.kmo_dummy + df_fl_add.kmo_headw)
        
        return df_tgb_rout_fl
    
    # %% calculate and correct sub-watershed polygons
    def calc_corr_sc_polyg(path_fd_c, path_tgb_p, val_no_dummy, def_a_min_tol,
                           path_gdb_out, name_tgb_sj='tgb_sj'):
        """
        This function calculates the final model cubcatchment watersheds and 
        corrects them concerning their geometrical mismatch resulting from
        the discretization of the digital elevation grid.
        Together with the D8-flow-direction, the watershed function allows
        multi-object shapes as they are logically linked as water flows 'over 
        the edge'. Anyhow, this results in duplicated ID numbers in the later
        process. Therefore, these (often small) polygons have to be merged into
        neighbour polygons.
    
        JM 2021
    
        Arguments:
        -----------
        path_fd_c: str
            path of the flow direction raster within model domain
        path_tgb_p: str
            path of point feature class containing final pour points of model
            elements
        val_no_dummy: int
            default value for no-dummy elements in variable 'tgb_dtgb' of 
            the tgb_p point feature class [-] (e.g., -1)
        def_a_min_tol: float
            minimum subcatchment area tolerated [m²] (e.g., 50)
        path_gdb_out: str
            path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
        name_tgb_sj: str (optional, default: 'tgb_sj')
            file name, where the feature class shall be stored
        
        Returns:
        -----------
        Saves an ArcGIS polygon feature class with the provided information
        """
        # paths for outputs
        path_tgb_sj         = path_gdb_out + name_tgb_sj
        # paths for intermediates
        path_tgb_r          = path_gdb_out + 'tgb_r'
        path_tgb_rg         = path_gdb_out + 'tgb_rg'
        path_tgb_nm         = path_gdb_out + 'tgb_nm'
        path_tgb_nr         = path_gdb_out + 'tgb_nr'
        path_tgb_pj         = path_gdb_out + 'tgb_pj'
        path_tgb_s          = path_gdb_out + 'tgb_s'
        path_tgb_s_del      = path_gdb_out + 'tgb_s_del'
        path_tgb_s_del_amax = path_gdb_out + 'tgb_s_del_amax'
        # internal field names    
        f_tgb      = 'tgb'
        f_tgb_type = 'tgb_type'
        f_kmu      = 'kmu'
        f_kmo      = 'kmo'
        f_tgb_dtgb = 'tgb_dtgb'
        # arcpy field names (ONLY CHANGE IF ARCPY FUNCTION CHANGES FORCE YOU!)
        f_r_val    = 'value'
        f_gc       = 'gridcode'
        f_ct       = 'Count'
        f_shp_a    = 'Shape_Area'
        # set environments
        arcpy.env.extent = 'MAXOF'
        arcpy.env.snapRaster = path_fd_c
        # calculate subcatchments based on pour points
        tgb_r = arcpy.sa.Watershed(path_fd_c, path_tgb_p, f_tgb)
        tgb_r.save(path_tgb_r)
        # convert raster to polygon
        arcpy.conversion.RasterToPolygon(
                path_tgb_r, path_tgb_s_del, 'NO_SIMPLIFY', f_r_val,
                'SINGLE_OUTER_PART', None)
        # calculate maximum single polygon area per TGB ID
        arcpy.analysis.Statistics(
                path_tgb_s_del, path_tgb_s_del_amax, '{0:s} MAX'.format(f_shp_a), f_gc)
        # join maximum polygon area to subcatchment polygons
        arcpy.management.JoinField(
                path_tgb_s_del, f_gc, path_tgb_s_del_amax, f_gc,
                'MAX_{0:s}'.format(f_shp_a))
        # select and delete polygons, where maximum area does not equal polygon area
        sel_expr = '{0:s} <> MAX_{0:s}'.format(f_shp_a)
        tgb_s_del_sel = arcpy.management.SelectLayerByAttribute(
                path_tgb_s_del, 'NEW_SELECTION', sel_expr, 'NON_INVERT')
        arcpy.management.DeleteFeatures(tgb_s_del_sel)
        arcpy.management.SelectLayerByAttribute(
                path_tgb_s_del, 'CLEAR_SELECTION', sel_expr, None)
        # convert subcatchment polygons to raster
        arcpy.conversion.PolygonToRaster(
                path_tgb_s_del, f_gc, path_tgb_nm, 'CELL_CENTER', 'NONE', path_fd_c)
        # set subcatchments null, which are below defined area threshold
        tgb_rg = arcpy.sa.RegionGroup(path_tgb_r, 'FOUR', 'WITHIN', 'ADD_LINK', None)
        tgb_rg.save(path_tgb_rg)
        gis_fd_c = arcpy.Raster(path_fd_c)
        cellsz = gis_fd_c.meanCellHeight
        tgb_nm = arcpy.ia.SetNull(path_tgb_rg, 1, '{0:s} < {1:.0f}'.format(
                f_ct, def_a_min_tol / (cellsz**2)))
        tgb_nm.save(path_tgb_nm)
        # nibble residual elements
        tgb_nr = arcpy.sa.Nibble(
                path_tgb_rg, path_tgb_nm, 'ALL_VALUES', 'PRESERVE_NODATA', None)
        tgb_nr.save(path_tgb_nr)
        # convert back to subcatchment polygons
        arcpy.conversion.RasterToPolygon(
                path_tgb_nr, path_tgb_s, 'NO_SIMPLIFY', f_r_val,
                'SINGLE_OUTER_PART', None)
        # delete subcatchments not nibbled (isolated subcatchments at the
        # overall watershed boundary)
        sel_expr = '{0:s} < {1:d}'.format(f_shp_a, def_a_min_tol)
        tgb_s_del_sel = arcpy.management.SelectLayerByAttribute(
                path_tgb_s, 'NEW_SELECTION', sel_expr, None)
        arcpy.management.DeleteFeatures(tgb_s_del_sel)
        arcpy.management.SelectLayerByAttribute(
                tgb_s_del_sel, 'CLEAR_SELECTION', sel_expr, None)    
        # select and copy points which are not representing dummy elements
        sel_expr = '{0:s} = {1:d}'.format(f_tgb_dtgb, val_no_dummy)
        tgb_pj = arcpy.management.SelectLayerByAttribute(
                path_tgb_p, 'NEW_SELECTION', sel_expr, None)
        arcpy.management.CopyFeatures(tgb_pj, path_tgb_pj, '', None, None, None)
        arcpy.management.SelectLayerByAttribute(
                tgb_pj, 'CLEAR_SELECTION', sel_expr, None)
        # join attributes to polygons
        arcpy.analysis.SpatialJoin(
                path_tgb_s, path_tgb_pj, path_tgb_sj,
                'JOIN_ONE_TO_ONE', 'KEEP_ALL', (
                        '{0:s} "{0:s}" true true false  4 Long 0 0,'
                            'First, #, {4:s}, {0:s}, -1, -1;'
                        '{1:s} "{1:s}" true true false 28 Text 0 0,'
                            'First, #, {4:s}, {1:s},  0, 28;'
                        '{2:s} "{2:s}" true true false  4 Long 0 0,'
                            'First, #, {4:s}, {2:s}, -1, -1;'
                        '{3:s} "{3:s}" true true false  4 Long 0 0,'
                            'First, #, {4:s}, {3:s}, -1, -1;').format(
                        f_tgb, f_tgb_type, f_kmu, f_kmo, path_tgb_p),
                'CONTAINS', None, '')
    
    # %% calculate subcatchment runoff concentration parameters
    def calc_sc_roconc_pars(path_dem_c, path_fl_c, path_tgb_p, path_tgb_sj, 
                            path_gdb_out, str_headw, str_routing):
        """
        This function calculates the necessary runoff concentration parameters
        and updates the model elements' point and polygon feature classes.
        The following parameters are calculated:
            - hot: upper elevation value for Kirpich concentration time estimation
            - tal: maximum flow length value for Kirpich concentration time estimation
    
        JM 2021
    
        Arguments:
        -----------
        path_dem_c: str
            path of the elevation raster within model domain
        path_fl_c: str
            path of the flow length raster within model domain
        path_tgb_p: str
            path of point feature class containing final pour points of model
            elements
        path_tgb_sj: str (optional, default: 'tgb_sj')
            path of polygon feature class containing final model elements'
            shapes including calculated parameters
        path_gdb_out: str
            path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
        str_headw, str_routing, str_dummy: str
            Strings representing the model elements' types in ser_tgb_type
            (e.g., 'headwater', 'routing', 'dummy')
        
        Returns:
        -----------
        Updates the ArcGIS polygon feature class with the provided information
        """
        def findField(fc, fi):
            """
            This function returns True if the feature classes attribute table
            contains a field with teh asked field name and False otherwise.
            """
            fieldnames = [field.name for field in arcpy.ListFields(fc)]
            if fi in fieldnames: return True
            else: return False
        
        # paths for intermediates
        path_tgb_zmax = path_gdb_out + 'tgb_zmax'
        path_tgb_flmax = path_gdb_out + 'tgb_flmax'
        # internal field names    
        f_tgb      = 'tgb'
        f_tgb_type = 'tgb_type'
        f_kmu      = 'kmu'
        f_kmo      = 'kmo'
        f_hot      = 'hot'
        f_tal      = 'tal'
        # arcpy field names (ONLY CHANGE IF ARCPY FUNCTION CHANGES FORCE YOU!)
        f_zs_max   = 'MAX'
        # set environments
        arcpy.env.extent = 'MAXOF'
        arcpy.env.snapRaster = path_dem_c
        # summarize maximum elevation and flow length rasters within subcatchments
        arcpy.sa.ZonalStatisticsAsTable(path_tgb_sj, f_tgb, path_dem_c, path_tgb_zmax,
                                        'DATA', 'MAXIMUM', 'CURRENT_SLICE')
        arcpy.sa.ZonalStatisticsAsTable(path_tgb_sj, f_tgb, path_fl_c, path_tgb_flmax,
                                        'DATA', 'MAXIMUM', 'CURRENT_SLICE')
        # join maximum elevation field and alter field name
        arcpy.management.JoinField(path_tgb_sj, f_tgb, path_tgb_zmax, f_tgb, f_zs_max)
        if not findField(path_tgb_sj, f_hot):
            arcpy.management.AlterField(
                    path_tgb_sj, f_zs_max, f_hot,
                    '', 'DOUBLE', 8, 'NULLABLE', 'CLEAR_ALIAS')
        # join maximum flow length field
        arcpy.management.JoinField(path_tgb_sj, f_tgb, path_tgb_flmax, f_tgb, f_zs_max)
        # calculate concentration path length for routing elements
        # using max(flow length) - low boundary (= mean river flow length)
        sel_expr = "{0:s} = '{1:s}'".format(f_tgb_type, str_routing)
        tgb_sj_rout_sel = arcpy.management.SelectLayerByAttribute(
                path_tgb_sj, 'NEW_SELECTION', sel_expr, None)
        calc_expr = '(!{0:s}!-(!{1:s}!+!{2:s}!)/2)/1000'.format(f_zs_max, f_kmo, f_kmu)
        arcpy.management.CalculateField(tgb_sj_rout_sel, f_tal, calc_expr,
                                        'PYTHON3', '', 'DOUBLE')
        arcpy.management.SelectLayerByAttribute(
                tgb_sj_rout_sel, 'CLEAR_SELECTION', sel_expr, None)
        # calculate concentration path length for headwater elements
        # using max(flow length) - low boundary (= minimum river flow length)
        sel_expr = "{0:s} = '{1:s}'".format(f_tgb_type, str_headw)
        tgb_sj_head_sel = arcpy.management.SelectLayerByAttribute(
                path_tgb_sj, 'NEW_SELECTION', sel_expr, None)
        calc_expr = '(!{0:s}!-!{1:s}!)/1000'.format(f_zs_max, f_kmu)
        arcpy.management.CalculateField(tgb_sj_head_sel, f_tal, calc_expr,
                                        'PYTHON3', '', 'DOUBLE')
        arcpy.management.SelectLayerByAttribute(
                tgb_sj_head_sel, 'CLEAR_SELECTION', sel_expr, None)
        # join calculated fields
        join_expr = '{0:s};{1:s}'.format(f_hot, f_tal)
        arcpy.management.JoinField(path_tgb_p, f_tgb, path_tgb_sj, f_tgb, join_expr)
    
    # %% calculate coordinates
    def calc_sc_coords(path_tgb_p, path_tgb_sj, path_gdb_out):
        """
        This function calculates the coordinates of the model elements' centroids
        for all headwater and routing elements. For dummy elements, the pour point
        coordinates are used. Finally, the model elements' point and polygon 
        feature classes are updated.
        The coordinate system is taken from the model element feature classes.
        The following variables are calculated:
            - x: x-coordinate
            - y: y-coordinate
    
        JM 2021
    
        Arguments:
        -----------
        path_tgb_p: str
            path of point feature class containing final pour points of model
            elements
        path_tgb_sj: str (optional, default: 'tgb_sj')
            path of polygon feature class containing final model elements'
            shapes including calculated parameters
        path_gdb_out: str
            path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
        
        Returns:
        -----------
        Updates the ArcGIS polygon feature class with the provided information
        """
        # paths for intermediates
        path_tgb_sj_midp = path_gdb_out + 'tgb_sj_midp'
        # internal field names    
        f_tgb      = 'tgb'
        f_tgb_type = 'tgb_type'
        f_x        = 'x'
        f_y        = 'y'
        # arcpy field names (ONLY CHANGE IF ARCPY FUNCTION CHANGES FORCE YOU!)
        f_p_x      = 'POINT_X'
        f_p_y      = 'POINT_Y'
        f_shp_a    = 'Shape_Area'
        # calculate centroids of element polygons and get coordinates
        arcpy.management.FeatureToPoint(path_tgb_sj, path_tgb_sj_midp, 'CENTROID')
        arcpy.management.AddXY(path_tgb_sj_midp)
        # join coordinates to element points and alter field names
        arcpy.management.JoinField(path_tgb_p, f_tgb, path_tgb_sj_midp, f_tgb,
                                   '{0:s};{1:s}'.format(f_p_x, f_p_y))
        arcpy.management.CalculateField(
                path_tgb_p, f_x, '!{0:s}!'.format(f_p_x),
                'PYTHON3', '', 'DOUBLE')
        arcpy.management.CalculateField(
                path_tgb_p, f_y, '!{0:s}!'.format(f_p_y),
                'PYTHON3', '', 'DOUBLE')
        # add outflow point coordinates and use these for dummy elements
        arcpy.management.AddXY(path_tgb_p)
        sel_expr = "{0:s} = '{1:s}'".format(f_tgb_type, str_dummy)
        tgb_p_dummy_sel = arcpy.management.SelectLayerByAttribute(
                path_tgb_p, 'NEW_SELECTION', sel_expr, None)
        arcpy.management.CalculateField(
                tgb_p_dummy_sel, f_x, '!{0:s}!'.format(f_p_x),
                'PYTHON3', '', 'DOUBLE')
        arcpy.management.CalculateField(
                tgb_p_dummy_sel, f_y, '!{0:s}!'.format(f_p_y),
                'PYTHON3', '', 'DOUBLE')
        arcpy.management.SelectLayerByAttribute(
                path_tgb_p, 'CLEAR_SELECTION', sel_expr, None)
        # delete fields which are not necessary
        arcpy.management.DeleteField(
                path_tgb_p, '{0:s};{1:s};{2:s}'.format(f_p_x, f_p_y, f_shp_a))
    
    # %% fit tripel trapezoid to the mean of multiple derived cross sections
    def fit_mean_ttp(ser_pef_hm, ser_pef_bm, ser_pp_row, ser_pp_col, ser_pp_fa, ser_tgb_q_in,
                     dem, fd, cellsz,
                     gis_val_rows, gis_val_cols, gis_val_down_rows, gis_val_down_cols,
                     ser_tgb_up_nd, ser_tgb_type, str_routing,
                     def_cs_dist_eval=50, 
                     def_cs_wmax_eval=600, def_cs_hmax_eval=10,
                     def_flpl_wmax_eval=50,
                     def_ch_wmax_eval=40, def_ch_hmin_eval=0.1, def_ch_hmax_eval=3.0,
                     def_ch_hmin=0.2, def_ch_vmin=0.5, def_ch_vmax=3.0,
                     def_chbank_slmin=0.1,
                     def_val_wres=10.0, def_flpl_wres=0.20, 
                     def_ch_w=0.5, def_ch_h=0.5, def_ch_wres=0.05,
                     def_lam_hres=0.1, 
                     print_out=False,
                     ctrl_show_plots=False, ctrl_save_plots=False,
                     ser_tgb_a=None, ser_tgb_a_in=None, 
                     path_plots_out=None):
        
#                ser_pef_hm = df_ttp.loc[:, f_hm]
#                ser_pef_bm = df_ttp.loc[:, f_bm]
#                ser_pp_row = df_pp_sc.loc[:, f_row]
#                ser_pp_col = df_pp_sc.loc[:, f_col]
#                ser_pp_fa = df_pp_sc.loc[:, f_fa]
#                ser_tgb_q_in = df_tgb.loc[:, f_qin]
#                ser_tgb_type = df_tgb.loc[:, f_tgb_type]
#                ser_tgb_a=df_tgb.loc[:, f_ft]
#                ser_tgb_a_in=df_tgb.loc[:, f_ain]

        """
        This function estimates cross sections from a high-resolution digital
        elevation grid. For each model element, multiple cross sections are derived
        from the grid perpendicular to the flow accumulation path network. Every 
        cross section is corrected and finally merged by taking the mean such that
        only one mean cross section is left for each model element. Finally, the 
        default triple trapezoid function of LARSIM is fitted on the profile and 
        all cross section parameters for the model are calculated from it.
    
        JM 2021
    
        Arguments:
        -----------
        ser_pef_bm, ser_pef_hm: pandas.Series
            Series of estimated channel width ('bm') and depth ('wm')
        ser_pp_row, ser_pp_col: pandas.Series
            pour point row (ser_pp_row) and column (ser_pp_col) indices in 
            digital elevation array
        ser_pp_fa: pandas.Series
            pour point flow accumulation raster value
        ser_tgb_q_in: pandas.Series
            Series of elements' channel-forming discharge at the corresponding
            model element ID in the serie's index. 
        dem: numpy.array [m]
            Digital elevation grid represented as 2D numpy array.
        fd: numpy.array [m]
            Flow direction grid represented as 2D numpy array.
        cellsz: float [m]
            Cell size of the digital elevation grid
        gis_val_rows, gis_val_cols: numpy.array (vector) [-]
            Row and column array indices of elevation grid values, which are 
            within the model domain and at locations with a flow accumulation 
            larger than the user defined threshold for subcatchment creation.
        gis_val_down_rows, gis_val_down_cols: numpy.array (vector) [-]
            Row and column array indices of elevation grid values, which are 
            within the model domain, at locations with a flow accumulation 
            larger than the user defined threshold for subcatchment creation, 
            and which are the next downstream neighbour cells following the flow
            direction raster.
        ser_tgb_up_nd: pandas.Series
            Series of corresponding upstream model element indices ignoring dummy
            elements. These are represented as empty array (e.g., []).
        ser_tgb_type: pandas.Series
            Boolean Series, which identifies the headwater cells corresponding to the
            serie's ascending index with True.
            (e.g., pd.Series(data=[1, 1, 0, 0], index=[1, 2, 3, 4], name='J_type'))
        str_routing: str [-]
            string representing the model element type 'routing' element. 
            (e.g., 'routing')
        def_cs_dist_eval: int (optional, default: 50) [m]
            Linear distance between two automatically derived, consecutive 
            cross sections. This value should be oriented at the desired size
            of subcatchments to generate at least few cross sections.
        def_cs_wmax_eval: int (optional, default: 600) [m]
            Length of the automatically generated cross sections perpendicular
            to the flow accumulation flow network. It should cover the valley
            at least until the estimated maximum water depth.
        def_cs_hmax_eval: float (optional, default: 10) [m]
            maximum height of cross section evaluation
        def_flpl_wmax_eval: float (optional, default: 50) [m]
            estimated maximum flood plain width
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
        def_val_wres: float (optional, default: 10 [m])
            resolution of interpolated points within vally
        def_flpl_wres: float (optional, default: 0.20 [m])
            resolution of interpolated points within flood plain
        def_ch_w, def_ch_h: float (optional, default: 0.5, 0.5) [m]
            artificial channel width (w) and depth (h), added to continuiously
            descending cross sections
        def_ch_wres: float (optional, default: 0.05 [m])
            resolution of interpolated points within channel
        def_lam_hres: float (optional, default: 0.1) [m]
            spacing between evaluation lamellae
        print_out: boolean (optional, default: False)
            true if workprogress shall be print to command line
        ctrl_show_plots: boolean (optional, default: False) [-]
            (de-)activate pop-up of figures
        ctrl_save_plots: boolean (optional, default: False) [-]
            (de-)activate export of figures as files
        ser_tgb_a: pandas.Series (optional, default: None) [km²]
            model element subcatchment area
        ser_tgb_a_in: pandas.Series (optional, default: None) [km²]
            sum of upstream model elements' area
        path_plots_out: str (optional, default: None)
            path where plots are stored (e.g., 'c:\model_creation\fig')
        
        Returns:
        -----------
        df_ttp_fit: pandas.DataFrame
            DataFrame of triple trapezoid river cross section profile parameters.
            The DataFrame includes the model element ID as index and the following columns:
            - 'hm': channel depth [m]
            - 'bm': channel width [m]
            - 'bl': flat foreland width left [m]
            - 'br': flat foreland width right [m]
            - 'bbl': slopy foreland width left [m]
            - 'bbr': slopy foreland width right [m]
            - 'bnm': channel embankment slope left and right [mL/mZ]
            - 'bnl': slopy foreland slope left [mL/mZ]
            - 'bnr': slopy foreland slope right [mL/mZ]
            - 'bnvrl': outer foreland slope left [mL/mZ]
            - 'bnvrr': outer foreland slope right [mL/mZ]
        ser_ttp_fit_log: pandas.Series
            Series containing a log for each model element, which states 'fit'
            if the fitting was successful and 'est' if it was not and estimated
            parameters have to be used instead.
        df_ttp_fit_r2: pandas.DataFrame
            DataFrame containing the R² values as quality indicator for the
            fitting for each model element. It includes the following columns:
            - 'ch' : fitting of channel
            - 'fll': fitting of foreland left
            - 'flr': fitting of foreland right
        """
        
        # %% get array indices of recent and upstream routing elements using max. flow acc.
        def get_rout_arr_idx(ser_pp_row, ser_pp_col, ser_pp_fa,
                             ser_tgb_up_nd, ser_tgb_type, str_routing):
            """
            This function finds the array indices of the routing model elements
            and their upstream neighbours with the largest flow accumulation
            referencing into the flow direction raster array.
            
            JM 2021
            
            Arguments:
            -----------
            ser_pp_row, ser_pp_col: pandas.Series
                pour point row (ser_pp_row) and column (ser_pp_col) indices in 
                digital elevation array
            ser_pp_fa: pandas.Series
                pour point flow accumulation raster value
            ser_tgb_up_nd: pandas.Series
                Series of corresponding upstream model element indices ignoring dummy
                elements. These are represented as empty array (e.g., []).
            ser_tgb_type: pandas.Series
                Boolean Series, which identifies the headwater cells corresponding to the
                serie's ascending index with True.
                (e.g., pd.Series(data=[1, 1, 0, 0], index=[1, 2, 3, 4], name='J_type'))
            str_routing: str [-]
                string representing the model element type 'routing' element. 
                (e.g., 'routing')
            
            Returns:
            -----------
            df_tgb_rout_rc, df_tgb_up_rout_rc: pandas.DataFrame
                DataFrame of array indices into the flow direction array for recent 
                (df_tgb_rout_rc) and upstream elements (df_tgb_up_rout_rc).
                Each DataFrame includes the model element ID as index and the following columns:
                - 'row': row index of array [-]
                - 'col': column index of array [-]
            """
            # define internal variable names
            f_row = 'row'
            f_col = 'col'
            # get routing element indices
            tgb_type_rout_id = ser_tgb_type.index[ser_tgb_type == str_routing]
            # get upstream real element's array indices
            df_tgb_rout_rc    = pd.DataFrame(np.zeros((tgb_type_rout_id.shape[0], 2)) - 1, 
                                           index=tgb_type_rout_id,
                                           columns=[f_row, f_col]).astype(np.int)
            df_tgb_up_rout_rc = pd.DataFrame(np.zeros((tgb_type_rout_id.shape[0], 2)) - 1, 
                                           index=tgb_type_rout_id,
                                           columns=[f_row, f_col]).astype(np.int)
            for _, tgb in enumerate(tgb_type_rout_id.values):
                # get array indices of routing elements
                df_tgb_rout_rc.at[tgb] = np.hstack((ser_pp_row.at[tgb], ser_pp_col.at[tgb]))
                # get indices of routing upstream elements
                tgb_up_real = ser_tgb_up_nd.loc[tgb]
                # get flow accumulation of upstream elements
                tgb_up_max_fa = ser_pp_fa.loc[tgb_up_real].idxmax()
                # save upstream element's array indices
                df_tgb_up_rout_rc.at[tgb] = np.hstack((
                        ser_pp_row.loc[tgb_up_max_fa],
                        ser_pp_col.loc[tgb_up_max_fa]))
            return df_tgb_rout_rc, df_tgb_up_rout_rc
        
        # %% get cross section digital elevation grid profiles
        def get_cs_dem_prof(dem, fd, cellsz,
                            gis_val_rows, gis_val_cols,
                            gis_val_down_rows, gis_val_down_cols,
                            df_tgb_rout_rc, df_tgb_up_rout_rc, 
                            ser_tgb_type, str_routing,
                            def_cs_dist_eval=50, def_cs_wmax_eval=600,
                            print_out=False):
            """
            This function calculates digital elevation grid profiles normal to the
            flow accumulation flow path using a user defined linear distance and 
            profile width. Additionally, some corrections are performed:
                1. The profile is clipped at the first NaN values equally to both
                   sides starting from the flow path.
                2. The profiles' elevation is normalized to the flow path's
                   elevation, which is zero in the output.
            
            JM 2021
            
            Arguments:
            -----------
            dem: numpy.array [m]
                Digital elevation grid represented as 2D numpy array.
            fd: numpy.array [m]
                Flow direction grid represented as 2D numpy array.
            cellsz: float [m]
                Cell size of the digital elevation grid
            gis_val_rows, gis_val_cols: numpy.array (vector) [-]
                Row and column array indices of elevation grid values, which are 
                within the model domain and at locations with a flow accumulation 
                larger than the user defined threshold for subcatchment creation.
            gis_val_down_rows, gis_val_down_cols: numpy.array (vector) [-]
                Row and column array indices of elevation grid values, which are 
                within the model domain, at locations with a flow accumulation 
                larger than the user defined threshold for subcatchment creation, 
                and which are the next downstream neighbour cells following the flow
                direction raster.
            df_tgb_rout_rc, df_tgb_up_rout_rc: pandas.DataFrame [-]
                DataFrame of array indices into the flow direction array for recent 
                (df_tgb_rout_rc) and upstream elements (df_tgb_up_rout_rc).
                Each DataFrame includes the model element ID as index and the following columns:
                - 'row': row index of array [-]
                - 'col': column index of array [-]
            ser_tgb_type: pandas.Series [-]
                Series of model elements' types corresponding to the ascending index. 
                (e.g., pd.Series(['headwater', 'routing', 'headwater', 'routing'],
                                 index=[1, 2, 3, 4], name='tgb_type'))
            str_routing: str [-]
                string representing the model element type 'routing' element. 
                (e.g., 'routing')
            def_cs_dist_eval: int (optional, default: 50) [m]
                Linear distance between two automatically derived, consecutive 
                cross sections. This value should be oriented at the desired size
                of subcatchments to generate at least few cross sections.
            def_cs_wmax_eval: int (optional, default: 600) [m]
                Length of the automatically generated cross sections perpendicular
                to the flow accumulation flow network. It should cover the valley
                at least until the estimated maximum water depth.
            print_out: boolean (optional, default: False)
                true if workprogress shall be print to command line
            
            Returns:
            -----------
            ser_cs_l, ser_cs_h: pandas.Series
                Series of cross sections' distance (ser_cs_l) and elevation
                difference (ser_cs_h) from flow accumulation's flow paths.
            """
            # pre-define necessary functions
            def calc_buf_arr(pt_dem_x, pt_dem_y, dem_cell_x, dem_cell_y,
                             buf_size_x_1d, buf_size_y_1d):
                # calculate buffer size for both array sides
                buf_size_x = buf_size_x_1d * 2 + 1
                buf_size_y = buf_size_y_1d * 2 + 1
                # create raster array
                buf_arr_y, buf_arr_x = np.mgrid[
                    0 : buf_size_y : dem_cell_x,
                    0 : buf_size_x : dem_cell_y]
                # calculate coordinates
                buf_arr_x = buf_arr_x - buf_size_x_1d + pt_dem_x
                buf_arr_y = np.flipud(buf_arr_y - buf_size_y_1d + pt_dem_y)
                return buf_arr_x, buf_arr_y
            
            def calc_eukl_dist(vec_l_xy, pt_xy, buf_arr_x, buf_arr_y):
                buf_arr_yx_flat = np.transpose(np.vstack((buf_arr_x.flatten(),
                                                          buf_arr_y.flatten())))
                arr_cross = np.cross((vec_l_xy), (pt_xy - buf_arr_yx_flat))
                l_buf_eukl_dist = np.abs(arr_cross) / np.linalg.norm(vec_l_xy)
                l_buf_eukl_dist = np.reshape(l_buf_eukl_dist, buf_arr_x.shape)
                return l_buf_eukl_dist
            
            # calculation section
            # define internal variable names
            f_csl = 'cs_l'
            f_csh = 'cs_h'
            # (de-)activate command line output
            debug = False # (False/True)
            # get routing element indices
            tgb_type_rout_id = ser_tgb_type.index[ser_tgb_type == str_routing]
            # pre-define series
            ser_cs_l = pd.Series(tgb_type_rout_id.shape[0] * [[]],
                                 index=tgb_type_rout_id, name=f_csl)
            ser_cs_h = pd.Series(tgb_type_rout_id.shape[0] * [[]],
                                 index=tgb_type_rout_id, name=f_csh)
            # initialize counters
            count_tgb = 1
            count_sect = 1
            # iterate elements loading their upstream array indices
            for tgb, (tgb_up_row, tgb_up_col) in df_tgb_up_rout_rc.iterrows():
                # get cell ID lists for subcatchment
                ii = 0
                if debug and print_out:
                    print(tgb, count_tgb, tgb_up_col, tgb_up_row, end=': ')
                # pre-define collection lists
                cs_h_tgb = []
                cs_l_tgb = []
                cs_hr_tgb = []
                # get flow direction
                tgb_fd = fd[tgb_up_row, tgb_up_col]
                # initialize distance calculation
                row = copy.deepcopy(tgb_up_row)
                col = copy.deepcopy(tgb_up_col)
                # iterate as long as flow direction is not outside of model domain
                while not np.isnan(tgb_fd):
                    # initialize euklidian distance values
                    dx = 0
                    dy = 0
                    ctrl_rem_cs = False
                    # go downstream along flow direction as long as distance is
                    # smaller then threshold
                    while np.sqrt(dx ** 2 + dy ** 2) < def_cs_dist_eval:
                        # find index of cell in non-nan list
                        cell_ii = np.nonzero(np.all((gis_val_rows == row,
                                                     gis_val_cols == col), 0))
                        # get downstream row and column
                        col = gis_val_down_cols[cell_ii]
                        row = gis_val_down_rows[cell_ii]
                        # if catchment outflow is reached, break iteration
                        if row.shape[0] == 0:
                            if debug and print_out:
                                print('-> end of catchment')
                            ctrl_rem_cs = True
                            break
                        # if end of element's routing section is reached, break iteration
                        if np.any(np.all(df_tgb_rout_rc == np.hstack((row, col)), 1)):
                            if debug and print_out:
                                print('-> end of routing section')
                            ctrl_rem_cs = True
                            break
                        # calculate euklidian distance
                        dx = (tgb_up_row - row) * cellsz
                        dy = (tgb_up_col - col) * cellsz
                    # if it is not the remaining routing section upstream of next element
                    if not ctrl_rem_cs:
                        # get evaluation point coordinates
                        # alternatively one could use the midpoint of the section:
                        #     np.round((tgb_up_row + row) / 2, 0)[0]
                        #     np.round((tgb_up_col + col) / 2, 0)[0]
                        row_eval = tgb_up_row
                        col_eval = tgb_up_col
                        if debug and print_out:
                            print(count_sect, end=', ')
                        pt_xy = np.int64(np.hstack((col_eval, row_eval)))
                        # get downstream point coordinates
                        pt_down_xy = np.hstack((col, row))
                        # calculate direction vector of line segment between recent
                        # and downstream point
                        vec_l_xy = (pt_down_xy - pt_xy) * cellsz
                        # calculate distance from recent to downstream points
                        vec_l_length = np.around(np.linalg.norm(vec_l_xy), 6)
                        # calculate direction vector of line segment between recent
                        # and virtual cross section point
                        pt_buf_xy = pt_xy + np.flip(vec_l_xy) / vec_l_length * [
                                def_cs_wmax_eval / 2, -def_cs_wmax_eval / 2]
                        vec_buf_xy = pt_buf_xy - pt_xy
                        # calculate buffer size for one array side
                        buf_size_x_1d = np.int64(
                                np.round(0.25 * np.absolute(vec_l_xy[0]) \
                                         + np.absolute(vec_buf_xy[0]), 0))
                        buf_size_y_1d = np.int64(
                                np.round(0.25 * np.absolute(vec_l_xy[1]) \
                                         + np.absolute(vec_buf_xy[1]), 0))
                        # create raster array
                        buf_arr_x, buf_arr_y = np.int64(calc_buf_arr(
                                col_eval, row_eval, cellsz, cellsz,
                                buf_size_x_1d, buf_size_y_1d))
                        # calculate euklidian distance between array points and
                        # line segment
                        l_buf_eukl_dist = calc_eukl_dist(vec_l_xy, pt_xy,
                                                         buf_arr_x, buf_arr_y)
                        # calculate euklidian distance between array points and
                        # buffer direction (= normal to line direction)
                        rec_buf_eukl_dist = calc_eukl_dist(vec_buf_xy, pt_xy,
                                                           buf_arr_x, buf_arr_y)
                        # get logical array for half rectangle downstream and
                        # half rectangle upstream
                        buf_bool = np.logical_and(
                            rec_buf_eukl_dist <= 0.5 * cellsz,
                            l_buf_eukl_dist <= def_cs_wmax_eval/2)
                        # get dem cross section
                        h_cs = dem[buf_arr_y[buf_bool], buf_arr_x[buf_bool]]
                        # get distance vector
                        l_cs = l_buf_eukl_dist[buf_bool]
                        l_cs_center_ii = np.nonzero(l_cs == 0)[0][0]
                        l_cs[:l_cs_center_ii] = l_cs[:l_cs_center_ii] * (-1)
                        # sort values according to distances from channel
                        l_cs_sort_ii = np.argsort(l_cs)
                        # sort values according to distance
                        h_cs_sort = h_cs[l_cs_sort_ii]
                        l_cs_sort = l_cs[l_cs_sort_ii]
                        # clip cross section at first nan value from center
                        l_cs_sort_c = l_cs_sort.shape[0]
                        cs_mp_ii = np.int32((l_cs_sort_c-1)/2)
                        cs_nn_ii = np.zeros(h_cs_sort.shape, dtype=bool)
                        for ii, l in enumerate(l_cs_sort):
                            if cs_mp_ii > ii:
                                st = ii
                                end = cs_mp_ii
                            else:
                                st = cs_mp_ii
                                end = ii
                            if np.all(~np.isnan(h_cs_sort[st:end+1])):
                                cs_nn_ii[ii] = True
                        # append values to lists
                        cs_h_tgb.append(h_cs_sort[cs_nn_ii])
                        cs_l_tgb.append(l_cs_sort[cs_nn_ii])
                        # calculate elevation value at river point
                        cs_hr_tgb.append(h_cs[l_cs_center_ii])
                        # increment section counter
                        count_sect += 1
                        # initialize next iteration
                        tgb_up_row = copy.deepcopy(row)
                        tgb_up_col = copy.deepcopy(col)
                        # get flow direction
                        tgb_fd = fd[tgb_up_row, tgb_up_col]
                    else:
                        if debug and print_out:
                            print()
                        count_sect = 1
                        break
                # balance elevation of cross section by river intersection point
                for cs_h_ii, cs_h in enumerate(cs_h_tgb):
                    cs_h_dif = cs_hr_tgb[cs_h_ii]
                    cs_h_tgb[cs_h_ii] = cs_h - cs_h_dif
                # save to series
                ser_cs_h.at[tgb] = cs_h_tgb
                ser_cs_l.at[tgb] = cs_l_tgb
                # increment element counter
                count_tgb += 1
        
            return ser_cs_l, ser_cs_h
        
        # %% calculate mean cross section for each model element
        def calc_mean_cs(ser_cs_l, ser_cs_h,
                         def_cs_wmax_eval=600, def_flpl_wmax_eval=50, def_ch_wmax_eval=40, 
                         def_val_wres=10, def_flpl_wres=0.20,
                         def_ch_w=0.5, def_ch_h=0.5, def_ch_wres=0.05):
            """
            This function calculates a mean cross section out of all available 
            cross sections per model element. Additionally, few correction 
            algorithms are run on the profiles. The processing steps are:
                1. interpolate cross section values using linear interpolation
                2. fill NaN values of linearly interpolated values with nearest values
                3. calculate mean of cross sections within one model element
                4. clip cross section at both sides' elevation maxima
                5. add default channel, if cross section is monotonically decreasing
                    (i.e., has no identifiable channel)
           
            JM 2021
            
            Arguments:
            -----------
            ser_cs_l, ser_cs_h: pandas.Series
                Series of cross sections' distance (ser_cs_l) and elevation
                difference (ser_cs_h) from flow accumulation's flow paths.
            def_cs_wmax_eval: int (optional, default: 600) [m]
                Length of the automatically generated cross sections perpendicular
                to the flow accumulation flow network. It should cover the valley
                at least until the estimated maximum water depth.
            def_flpl_wmax_eval: float (optional, default: 50) [m]
                estimated maximum flood plain width
            def_ch_wmax_eval: float (optional, default: 40) [m]
                estimated maximum channel width
            def_val_wres: float (optional, default: 10 [m])
                resolution of interpolated points within vally
            def_flpl_wres: float (optional, default: 0.20 [m])
                resolution of interpolated points within flood plain
            def_ch_w, def_ch_h: float (optional, default: 0.5, 0.5) [m]
                artificial channel width (w) and depth (h), added to continuiously
                descending cross sections
            def_ch_wres: float (optional, default: 0.05 [m])
                resolution of interpolated points within channel
            
            Returns:
            -----------
            ser_cs_l_m, ser_cs_h_m: pandas.Series
                Series of mean cross sections' distance (ser_cs_l_m) and elevation
                difference (ser_cs_h_m) from flow accumulation's flow paths.       
            """
            # define interpolation vertices as distance from the flow network
            # define internal variable names
            f_cslm = 'cs_l_m'
            f_cshm = 'cs_h_m'
            # intersection point
            cs_l_r_tgb = np.hstack((
                    np.arange(-def_cs_wmax_eval / 2, -def_flpl_wmax_eval      , def_val_wres ),
                    np.arange(-def_flpl_wmax_eval  , -def_ch_wmax_eval        , def_flpl_wres),
                    np.arange(-def_ch_wmax_eval    ,  0                       , def_ch_wres  ),
                    np.arange( 0                   ,  def_ch_wmax_eval        , def_ch_wres  ),
                    np.arange( def_ch_wmax_eval    ,  def_flpl_wmax_eval      , def_flpl_wres),
                    np.arange( def_flpl_wmax_eval  ,  def_cs_wmax_eval / 2 + 1, def_val_wres )))
            # pre-define series for interpolated cross section points
            ser_cs_l_m = pd.Series(ser_cs_l.shape[0] * [[]],
                                     index=ser_cs_l.index, name=f_cslm)
            ser_cs_h_m = pd.Series(ser_cs_l.shape[0] * [[]],
                                     index=ser_cs_l.index, name=f_cshm)
            # iterate model elements and their associated cross section vertices
            for tgb, (cs_l_tgb, cs_h_tgb) in pd.concat((ser_cs_l, ser_cs_h), axis=1).iterrows():
                # if there is a cross section to interpolate, do calculation
                if len(cs_l_tgb) > 0:
                    # iterate cross sections and interpolate extracted cross section
                    # vertices at idealized cross section points
                    # pre-define list of elevation values
                    cs_h_r_tgb = []
                    # iterate cross sections of model element
                    for cs_l, cs_h in zip(cs_l_tgb, cs_h_tgb):
                        # interpolate values using linear interpolation method
                        cs_h_r_tgb_l = griddata(cs_l, cs_h, cs_l_r_tgb, 'linear')
                        # interpolate values using nearest interpolation method
                        cs_h_r_tgb_n = griddata(cs_l, cs_h, cs_l_r_tgb, 'nearest')
                        # determine NaN values in linearly interpolated values
                        isnan_ii = np.isnan(cs_h_r_tgb_l)
                        # fill NaN values in linear interpolation with nearest values
                        cs_h_r_tgb_l[isnan_ii] = cs_h_r_tgb_n[isnan_ii]
                        # save interpolated elevation values to list
                        cs_h_r_tgb.append(cs_h_r_tgb_l)
                        
                    # calculate mean of cross sections within model element
                    cs_h_r_mean_tgb = np.mean(np.array(cs_h_r_tgb), 0)
                    # find position of intersection point with flow network
                    cs_l_n_ii = np.argwhere(cs_l_r_tgb == 0)[0][0]
                    # split mean cross section in two parts (left and right)
                    cs_h_rl = cs_h_r_mean_tgb[:cs_l_n_ii]
                    cs_h_rr = cs_h_r_mean_tgb[cs_l_n_ii:]
                    cs_l_rl = cs_l_r_tgb[:cs_l_n_ii]
                    cs_l_rr = cs_l_r_tgb[cs_l_n_ii:]
                    # clip mean cross section at both sides' elevation maxima and
                    # merge cross section parts (left and right)
                    cs_h_maxl = np.argmax(cs_h_rl)
                    cs_h_maxr = np.argmax(cs_h_rr)
                    cs_h_rc_tgb = np.hstack((cs_h_rl[cs_h_maxl:], cs_h_rr[:cs_h_maxr]))
                    cs_l_rc_tgb = np.hstack((cs_l_rl[cs_h_maxl:], cs_l_rr[:cs_h_maxr]))
                    # if cross section is continuatly descending at left side,
                    # add default channel at left side
                    if np.all(cs_l_rc_tgb <= 0):
                        if cs_l_rc_tgb[-1] < 0: # if zero is not included in cross section
                            cs_l_rc_add = np.arange(0, def_ch_w, def_ch_wres)
                            cs_h_rc_add = np.arange(0, def_ch_h, def_ch_wres)                            
                        else: # if zero is included in cross section
                            cs_l_rc_add = np.arange(def_ch_wres, def_ch_w, def_ch_wres)
                            cs_h_rc_add = np.arange(def_ch_wres, def_ch_h, def_ch_wres)                            
                        cs_l_rc_tgb = np.hstack((cs_l_rc_tgb, cs_l_rc_add))
                        cs_h_rc_tgb = np.hstack((cs_h_rc_tgb, cs_h_rc_add))
                    # if cross section is continuatly descending at right side,
                    # add default channel at the right side
                    if np.all(cs_l_rc_tgb >= 0):
                        if cs_l_rc_tgb[0] > 0: # if zero is not included in cross section
                            cs_l_rc_add = np.arange(0, def_ch_w, def_ch_wres)
                            cs_h_rc_add = np.arange(0, def_ch_h, def_ch_wres)                            
                        else: # if zero is included in cross section
                            cs_l_rc_add = np.arange(def_ch_wres, def_ch_w, def_ch_wres)
                            cs_h_rc_add = np.arange(def_ch_wres, def_ch_h, def_ch_wres)                            
                        cs_l_rc_tgb = np.hstack((cs_l_rc_add[::-1], cs_l_rc_tgb))
                        cs_h_rc_tgb = np.hstack((cs_l_rc_add[::-1], cs_h_rc_tgb))
                    # store variables
                    ser_cs_l_m.at[tgb] = np.transpose(cs_l_rc_tgb, 0)
                    ser_cs_h_m.at[tgb] = np.transpose(cs_h_rc_tgb, 0)
                # if there is no cross section for model element, store empty arrays
                else:
                    ser_cs_l_m.at[tgb] = np.array([])
                    ser_cs_h_m.at[tgb] = np.array([])
            return ser_cs_l_m, ser_cs_h_m
                
        # %% fit channel and foreland
        def fit_ttp(ser_cs_l_m, ser_cs_h_m,
                    ser_ch_wsll_fit, ser_ch_wslr_fit, ser_ch_h_fit, 
                    def_cs_hmax_eval=10,
                    print_out=False,
                    ctrl_show_plots=False, ctrl_save_plots=False,
                    ser_cs_l=None, ser_cs_h=None,
                    ser_tgb_a=None, ser_tgb_a_in=None, ser_tgb_q_in=None,
                    def_cs_wmax_eval=None, path_plots_out=None):
            """
            This function fits triple trapezoid parameters to the mean cross section
            taking into account fit channel width and depth.
            
            Arguments:
            -----------
            ser_cs_l_m, ser_cs_h_m: pandas.Series
                Series of mean cross sections' distance (ser_cs_l_m) and elevation
                difference (ser_cs_h_m) from flow accumulation's flow paths.
            ser_ch_wsll_fit, ser_ch_wslr_fit, ser_ch_h_fit: pandas.Series
                Series containing the fit left ('ser_ch_wsll_fit') and right 
                ('ser_ch_wslr_fit') water surface levels as well as the depth 
                ('ser_ch_h_fit') for each model element.
            def_cs_hmax_eval: float (optional, default: 10) [m]
                maximum height of cross section evaluation
            print_out: boolean (optional, default: False)
                true if workprogress shall be print to command line
            ctrl_show_plots: boolean (optional, default: False) [-]
                (de-)activate pop-up of figures
            ctrl_save_plots: boolean (optional, default: False) [-]
                (de-)activate export of figures as files
            
            The following arguments are only required for plotting (if either
            ctrl_show_plots or ctrl_save_plots or both is/are True):
            
            ser_cs_l, ser_cs_h: pandas.Series (optional, default: None)
                Series of cross sections' distance (ser_cs_l) and elevation
                difference (ser_cs_h) from flow accumulation's flow paths.
            ser_tgb_a: pandas.Series (optional, default: None) [km²]
                model element subcatchment area
            ser_tgb_a_in: pandas.Series (optional, default: None) [km²]
                sum of upstream model elements' area
            ser_tgb_q_in: pandas.Series (optional, default: None) [m³/s]
                sum of upstream model elements' river-forming discharge
            def_cs_wmax_eval: int (optional, default: None) [m]
                Length of the automatically generated cross sections perpendicular
                to the flow accumulation flow network. It should cover the valley
                at least until the estimated maximum water depth.
            path_plots_out: str (optional, default: None) [-]
                path where plots are stored (e.g., 'c:\model_creation\fig')
            
            Returns:
            -----------
            df_ttp_fit: pandas.DataFrame
                DataFrame of fit triple trapezoid river cross section profile parameters.
                The DataFrame includes the model element ID as index and the following columns:
                - 'hm': channel depth [m]
                - 'bm': channel width [m]
                - 'bl': flat foreland width left [m]
                - 'br': flat foreland width right [m]
                - 'bbl': slopy foreland width left [m]
                - 'bbr': slopy foreland width right [m]
                - 'bnm': channel embankment slope left and right [mL/mZ]
                - 'bnl': slopy foreland slope left [mL/mZ]
                - 'bnr': slopy foreland slope right [mL/mZ]
                - 'bnvrl': outer foreland slope left [mL/mZ]
                - 'bnvrr': outer foreland slope right [mL/mZ]
            ser_ttp_fit_log: pandas.Series
                Series containing a log for each model element, which states 'fit'
                if the fitting was successful and 'est' if it was not and estimated
                parameters have to be used instead.
            df_ttp_fit_r2: pandas.DataFrame
                DataFrame containing the R² values as quality indicator for the
                fitting for each model element. It includes the following columns:
                - 'ch' : fitting of channel
                - 'fll': fitting of foreland left
                - 'flr': fitting of foreland right
            """
            # channel fit function
            def ch_fit(x, x_bl_tmp, y_bl, x_wsl_l, y_wsl_l, x_bm_l, y_bm_l, 
                       x_bm_r, y_bm_r, x_wsl_r, y_wsl_r, x_br_tmp, y_br):
                ch_p = np.array([
                        [x_bl_tmp, y_bl   ],
                        [x_wsl_l,  y_wsl_l],
                        [x_bm_l,   y_bm_l ],
                        [x_bm_r,   y_bm_r ],
                        [x_wsl_r,  y_wsl_r],
                        [x_br_tmp, y_br   ]])
                return griddata(ch_p[:,0], ch_p[:,1], x, 'linear')
            # foreland fit function left
            def fll_fit(x, x_bnvrl, y_bnvrl, x_bbl, y_bbl, x_bl, y_bl,
                        x_wsl_l, y_wsl_l, x_bm_l, y_bm_l):
                ttp_p = np.array([
                        [x_bnvrl,  y_bnvrl],
                        [x_bbl,    y_bbl  ],
                        [x_bl,     y_bl   ],
                        [x_wsl_l,  y_wsl_l],
                        [x_bm_l,   y_bm_l ],
                        [0,        0      ]])
                return griddata(ttp_p[:,0], ttp_p[:,1], x, 'linear')
            # foreland fit function rigth
            def flr_fit(x, x_bm_r, y_bm_r, x_wsl_r, y_wsl_r, x_br, y_br,
                        x_bbr, y_bbr, x_bnvrr, y_bnvrr):
                ttp_p = np.array([
                        [0,        0      ],
                        [x_bm_r,   y_bm_r ],
                        [x_wsl_r,  y_wsl_r],
                        [x_br,     y_br   ],
                        [x_bbr,    y_bbr  ],
                        [x_bnvrr,  y_bnvrr]])
                return griddata(ttp_p[:,0], ttp_p[:,1], x, 'linear')
            
            # input error handling section
            if def_ch_hmin < 0.2: sys.exit('def_ch_hmin has to be >= 0.2')
            # predefine variables
            df_ttp_fit = pd.DataFrame(
                    np.zeros((ser_cs_l_m.shape[0], 11)) * np.nan,
                    index=ser_cs_l_m.index,
                    columns=['bm', 'hm', 'bnm', 'bl', 'br', 'bbl', 'bbr',
                             'bnl', 'bnr', 'bnvrl', 'bnvrr'])
            ser_ttp_fit_log = pd.Series(
                    ser_cs_l_m.shape[0]*[[]], index=ser_cs_l_m.index,
                    name='type')
            df_ttp_fit_r2 = pd.DataFrame(
                    np.zeros((ser_cs_l_m.shape[0], 3)) * np.nan,
                    index=ser_cs_l_m.index,
                    columns=['ch', 'fll', 'flr'])
            
            # iterate model elements' mean cross sections and fit channel values
            for tgb, (cs_l_m_tgb, cs_h_m_tgb, wsll_tgb, wslr_tgb, hm_tgb) \
                in pd.concat((ser_cs_l_m, ser_cs_h_m, ser_ch_wsll_fit, 
                              ser_ch_wslr_fit, ser_ch_h_fit), axis=1).iterrows():
                    # test if estimated water surface level is empty
                    if not np.any(np.isnan([wsll_tgb, wslr_tgb])):
                        
                        # channel fit
                        # clip mean cross section to channel section
                        ii_l = np.nonzero(cs_l_m_tgb == 0)[0][0]
                        while cs_h_m_tgb[ii_l] <= hm_tgb and ii_l > 0: 
                            ii_l -= 1
                        ii_r = np.nonzero(cs_l_m_tgb == 0)[0][0]
                        while cs_h_m_tgb[ii_r] <= hm_tgb and ii_r < cs_h_m_tgb.shape[0] - 1:
                            ii_r += 1
                        ch_cs_h = cs_h_m_tgb[ii_l : ii_r + 1]
                        ch_cs_l = cs_l_m_tgb[ii_l : ii_r + 1]
                        # make a model
                        model_ch = lmfit.Model(ch_fit)
                        # create parameters and set initial values
                        params_ch = lmfit.Parameters()
                        params_ch.add('x_bl_tmp', value=ch_cs_l[0],  vary=False)
                        params_ch.add('y_bl'    , value=hm_tgb,      vary=False)
                        params_ch.add('x_wsl_l' , value=ch_cs_l[0],  vary=True,
                                      min=ch_cs_l[0], max=0)
                        params_ch.add('y_wsl_l' , value=hm_tgb,      vary=False)
                        params_ch.add('x_bm_l'  , value=-1,          vary=True,
                                      min=ch_cs_l[0], max=0)
                        params_ch.add('y_bm_l'  , value=0,           vary=False)
                        params_ch.add('x_bm_r'  , value=1,           vary=True,
                                      min=0         , max=ch_cs_l[-1])
                        params_ch.add('y_bm_r'  , value=0,           vary=False)
                        params_ch.add('x_wsl_r' , value=ch_cs_l[-1], vary=True,
                                      min=0         , max=ch_cs_l[-1])
                        params_ch.add('y_wsl_r' , value=hm_tgb,      vary=False)
                        params_ch.add('x_br_tmp', value=ch_cs_l[-1], vary=False)
                        params_ch.add('y_br'    , value=hm_tgb,      vary=False)
                        # inequality constraints
                        params_ch.add('xdiff_bltmp_wsll', value=1, min=0)
                        params_ch['x_wsl_l'].expr = 'x_bl_tmp - xdiff_bltmp_wsll'
                        params_ch.add('xdiff_brtmp_wslr', value=1, min=0)
                        params_ch['x_wsl_r'].expr = 'x_br_tmp + xdiff_brtmp_wslr'
                        # perform the channel fit
                        result_ch = model_ch.fit(ch_cs_h, params_ch, x=ch_cs_l)
                        x_wsl_l   = result_ch.values['x_wsl_l']
                        x_bm_l    = result_ch.values['x_bm_l' ]
                        x_bm_r    = result_ch.values['x_bm_r' ]
                        x_wsl_r   = result_ch.values['x_wsl_r']
                        df_ttp_fit_r2.at[tgb, 'ch'] \
                            = 1 - np.sum((result_ch.residual - np.mean(result_ch.data)) ** 2) \
                                / np.sum((result_ch.data     - np.mean(result_ch.data)) ** 2)
                        
                        # foreland fit left
                        # clip mean cross section to left foreland section
                        ii_l = np.nonzero(cs_l_m_tgb == 0)[0][0]
                        ii_c = np.nonzero(cs_l_m_tgb == 0)[0][0]
                        h_l  = cs_h_m_tgb[ii_l]
                        while h_l <= def_cs_hmax_eval and ii_l > 0:
                            ii_l -= 1
                            h_l = cs_h_m_tgb[ii_l]
                        ch_cs_h = cs_h_m_tgb[ii_l : ii_c + 1]
                        ch_cs_l = cs_l_m_tgb[ii_l : ii_c + 1]
                        if np.min(ch_cs_l) >= x_wsl_l:
                            ch_cs_h = np.hstack((hm_tgb  + 0.1, ch_cs_h))
                            ch_cs_l = np.hstack((x_wsl_r - 1.0, ch_cs_l))
                        # get constraint values from cross section
                        bnvrl_lmax = ch_cs_l[0]
                        bnvrl_hmax = ch_cs_h[0]
                        # make a model
                        fll_model = lmfit.Model(fll_fit)
                        # create parameters and set initial values
                        params_fll = lmfit.Parameters()
                        params_fll.add('x_bnvrl', value=bnvrl_lmax, vary=False)
                        params_fll.add('y_bnvrl', value=bnvrl_hmax, vary=False)
                        params_fll.add('x_bbl'  , value=x_wsl_l   , vary=True,
                                       min=bnvrl_lmax + 0.01, max=x_wsl_l)
                        params_fll.add('y_bbl'  , value=hm_tgb    , vary=True,
                                       min=hm_tgb           , max=bnvrl_hmax + 0.01)
                        params_fll.add('x_bl'   , value=x_wsl_l   , vary=True,
                                       min=bnvrl_lmax       , max=x_wsl_l)
                        params_fll.add('y_bl'   , value=hm_tgb    , vary=False)
                        params_fll.add('x_wsl_l', value=x_wsl_l   , vary=False)
                        params_fll.add('y_wsl_l', value=hm_tgb    , vary=False)
                        params_fll.add('x_bm_l' , value=x_bm_l    , vary=False)
                        params_fll.add('y_bm_l' , value=0         , vary=False)
                        # inequality constraints
                        params_fll.add('xdiff_bl_wsll', value=1, min=0.01)
                        params_fll['x_bl' ].expr = 'x_wsl_l - xdiff_bl_wsll'
                        params_fll.add('xdiff_bbl_bl' , value=1, min=0.01)
                        params_fll['x_bbl'].expr = 'x_bl - xdiff_bbl_bl'
                        params_fll.add('ydiff_bbl_bl' , value=1, min=0.01)
                        params_fll['y_bbl'].expr = 'y_bl + ydiff_bbl_bl'
                        # perform the channel fit
                        result_fll = fll_model.fit(ch_cs_h, params_fll, x=ch_cs_l)
                        x_bbl = result_fll.values['x_bbl']
                        y_bbl = result_fll.values['y_bbl']
                        x_bl  = result_fll.values['x_bl' ]
                        df_ttp_fit_r2.at[tgb, 'fll'] \
                            = 1 - np.sum((result_fll.residual - np.mean(result_fll.data)) ** 2) \
                                / np.sum((result_fll.data     - np.mean(result_fll.data)) ** 2)
            
                        # foreland fit right
                        # clip mean cross section to left foreland section
                        ii_r = np.nonzero(cs_l_m_tgb == 0)[0][0]
                        h_r  = cs_h_m_tgb[ii_r]
                        while h_r <= def_cs_hmax_eval and ii_r < cs_h_m_tgb.shape[0] - 1:
                            ii_r += 1
                            h_r = cs_h_m_tgb[ii_r]
                        ch_cs_h = cs_h_m_tgb[ii_c : ii_r + 1]
                        ch_cs_l = cs_l_m_tgb[ii_c : ii_r + 1]
                        if np.max(ch_cs_l) <= x_wsl_r:
                            ch_cs_h = np.hstack((ch_cs_h, hm_tgb  + 0.1))
                            ch_cs_l = np.hstack((ch_cs_l, x_wsl_r + 1.0))
                        # get constraint values from cross section
                        bnvrr_lmax = ch_cs_l[-1]
                        bnvrr_hmax = ch_cs_h[-1]
                        # make a model
                        flr_model = lmfit.Model(flr_fit)
                        # create parameters and set initial values
                        params_flr = lmfit.Parameters()
                        params_flr.add('x_bm_r' , value=x_bm_r    , vary=False)
                        params_flr.add('y_bm_r' , value=0         , vary=False)
                        params_flr.add('x_wsl_r', value=x_wsl_r   , vary=False)
                        params_flr.add('y_wsl_r', value=hm_tgb    , vary=False)
                        params_flr.add('x_br'   , value=wslr_tgb  , vary=True,
                                       min=x_wsl_r, max=bnvrr_lmax)
                        params_flr.add('y_br'   , value=hm_tgb    , vary=False)
                        params_flr.add('x_bbr'  , value=wslr_tgb  , vary=True,
                                       min=x_wsl_r, max=bnvrr_lmax - 0.01)
                        params_flr.add('y_bbr'  , value=hm_tgb    , vary=True,
                                       min=hm_tgb  , max=bnvrr_hmax - 0.01)
                        params_flr.add('x_bnvrr', value=bnvrr_lmax, vary=False)
                        params_flr.add('y_bnvrr', value=bnvrr_hmax, vary=False)
                        # inequality constraints
                        params_flr.add('xdiff_br_wslr', value=1, min=0.01)
                        params_flr['x_br' ].expr = 'x_wsl_r + xdiff_br_wslr'
                        params_flr.add('xdiff_bbr_br' , value=1, min=0.01)
                        params_flr['x_bbr'].expr = 'x_br + xdiff_bbr_br'
                        params_flr.add('ydiff_bbr_br' , value=1, min=0.01)
                        params_flr['y_bbr'].expr = 'y_br + ydiff_bbr_br'
                        # perform the channel fit
                        result_flr = flr_model.fit(ch_cs_h, params_flr, x=ch_cs_l)
                        x_bbr = result_flr.values['x_bbr']
                        y_bbr = result_flr.values['y_bbr']
                        x_br  = result_flr.values['x_br' ]
                        df_ttp_fit_r2.at[tgb, 'flr'] = 1 \
                                - np.sum((result_flr.residual - np.mean(result_flr.data)) ** 2) \
                                / np.sum((result_flr.data     - np.mean(result_flr.data)) ** 2)
            
                        # calculate and summarize tripel trapezoid parameters
                        # summarize
                        wsl_tgb = np.abs(x_wsl_l) + x_wsl_r
                        bm_tgb  = np.abs(x_bm_l)  + x_bm_r
                        bbl_tgb = np.abs(x_bbl - x_bl)
                        bbr_tgb =        x_bbr - x_br
                        # channel
                        df_ttp_fit.at[tgb, 'bm'   ] = bm_tgb
                        df_ttp_fit.at[tgb, 'hm'   ] = hm_tgb
                        df_ttp_fit.at[tgb, 'bnm'  ] = (wsl_tgb - bm_tgb) / 2 / hm_tgb
                        # foreland left
                        df_ttp_fit.at[tgb, 'bl'   ] = np.abs(x_bl - x_wsl_l)
                        df_ttp_fit.at[tgb, 'bbl'  ] = bbl_tgb
                        df_ttp_fit.at[tgb, 'bnl'  ] = min(bbl_tgb / (y_bbl - hm_tgb), 999)
                        df_ttp_fit.at[tgb, 'bnvrl'] = min(np.abs(bnvrl_lmax - x_bbl) \
                                     / (bnvrl_hmax - y_bbl), 999)
                        # foreland right
                        df_ttp_fit.at[tgb, 'br'   ] = x_br - x_wsl_r
                        df_ttp_fit.at[tgb, 'bbr'  ] = bbr_tgb
                        df_ttp_fit.at[tgb, 'bnr'  ] = min(bbr_tgb / (y_bbr - hm_tgb), 999)
                        df_ttp_fit.at[tgb, 'bnvrr'] = min((bnvrr_lmax - x_bbr) \
                                     / (bnvrr_hmax - y_bbr), 999)
                        
                        # plot figure (if activated)
                        if ctrl_show_plots or ctrl_save_plots:
                            # turn plot visibility on or off
                            if ctrl_show_plots: plt.ion()
                            else:               plt.ioff()
                            # get cross sections
                            cs_h_tgb = ser_cs_h.at[tgb]
                            cs_l_tgb = ser_cs_l.at[tgb]
                            # calculate fitted cross section vertices
                            ttp_fit_p_tgb = np.array([
                                    [bnvrl_lmax, bnvrl_hmax],
                                    [x_bbl,      y_bbl     ],
                                    [x_bl,       hm_tgb    ],
                                    [x_wsl_l,    hm_tgb    ],
                                    [x_bm_l,     0         ],
                                    [x_bm_r,     0         ],
                                    [x_wsl_r,    hm_tgb    ],
                                    [x_br,       hm_tgb    ],
                                    [x_bbr,      y_bbr     ],
                                    [bnvrr_lmax, bnvrr_hmax]])
                            # interpolate vertices
                            int_l = np.arange(cs_l_m_tgb[0], cs_l_m_tgb[-1] + 0.1, 0.1)
                            ttp_fit_tgb = griddata(
                                    ttp_fit_p_tgb[: , 0], ttp_fit_p_tgb[: , 1],
                                    int_l, 'linear')
                            # define label size
                            l_sz = 14
                            # define colors
                            c_pl_all = (.6, .6, .6)
                            c_is_all = (.8, .8, .8)
                            c_wl = (.2, .2, 1)
                            c_surf = '#663300'
                            c_fit = (0, .5, 0)
                            # define line widths
                            lw_cs_all = 0.5
                            lw_pl = 1.5
                            lw_is = 1.0
                            # create figure
                            fig = plt.figure(figsize=[8, 5])
                            ax_pl_pos = [.09, .11, .88, .83] # position of ax_pl within fig
                            ax_is_pos = [.40, .77, .30, .20] # position of inset within ax_pl
                            ax_pl = fig.add_subplot(111, position=ax_pl_pos)
                            # plot all sample cross sections
                            for cs_ii, cs_l in enumerate(cs_l_tgb):
                                if cs_ii == 0: label_str='evaluated cross sections'
                                else: label_str='_nolegend_'
                                ax_pl.plot(cs_l, cs_h_tgb[cs_ii], color=c_pl_all,
                                        label=label_str, linewidth=lw_cs_all, zorder=1)
                            # plot mean cross section
                            ax_pl.plot(cs_l_m_tgb, cs_h_m_tgb, c=c_surf,
                                    label='averaged surface', linewidth=lw_pl, 
                                    zorder=5)
                            # plot fitted triple trapezoid profile and vertices
                            ax_pl.plot([-wsll_tgb, wslr_tgb], [hm_tgb, hm_tgb], c=c_wl,
                                    zorder=19, linewidth=lw_pl, label='bankful water level')
                            ax_pl.plot(int_l, ttp_fit_tgb, c=c_fit, ls='--',
                                    label='fitted TTP', zorder=20,
                                    linewidth=lw_pl)
                            ax_pl.scatter(ttp_fit_p_tgb[:, 0], ttp_fit_p_tgb[:, 1],
                                       s=40 , marker='o', facecolors='none',
                                       edgecolors=c_fit, zorder=50, linewidths=lw_pl,
                                       label='vertices of fitted TTP')
                            plt_str = ('model element: {0:d}\nA element: {1:.2f} km²\n'
                                       'A inflow: {2:.1f} km²\nQ inflow: {3:.1f} m³/s').format(
                                               tgb, ser_tgb_a.at[tgb],
                                               ser_tgb_a_in.at[tgb], ser_tgb_q_in.at[tgb])
                            ax_pl.text(def_cs_wmax_eval / 2 * 0.97, -0.7, plt_str,
                                    fontsize=l_sz - 2, ha='right', va='baseline')
                            # add legend, set axis limits and add labels and title
                            ax_pl.legend(loc='lower left', fontsize=l_sz - 2, labelspacing=.2)
                            ax_pl.set_xlim(-def_cs_wmax_eval / 2, def_cs_wmax_eval / 2  )
                            ax_pl.set_ylim(-1                   , 1.1 * def_cs_hmax_eval)
                            plt.xticks(fontsize=l_sz)
                            plt.yticks(fontsize=l_sz)
                            plt.xlabel('width [m]' , fontsize=l_sz)
                            plt.ylabel('height [m]', fontsize=l_sz)
                            plt.title('mean and fitted triple trapezoid profiles', fontsize=l_sz)
                            
                            # Create a set of inset Axes: these should fill the bounding box allocated to
                            # them.
                            ax_is = plt.axes([0, 0, 1, 1], zorder=100)
                            # Manually set the position and relative size of the inset axes within ax1
                            ip = InsetPosition(ax_pl, ax_is_pos)
                            ax_is.set_axes_locator(ip)
                            # Mark the region corresponding to the inset axes on ax1 and draw lines
                            # in grey linking the two axes.
                            pp, p1, p2 = mark_inset(ax_pl, ax_is, loc1=1, loc2=2,
                                fc="none", lw=.7, ec='k', ls='-', zorder=80)
                            ax_is.set_xticks([])
                            ax_is.set_yticks([])
                            ax_pl.set_zorder(90)
                            ax_is.set_zorder(200)
                            ax_ispl = plt.axes([0, 0, 1, 1], zorder=300, position=[
                                    ax_pl_pos[0]+ax_is_pos[0]*ax_pl_pos[2],
                                    ax_pl_pos[1]+ax_is_pos[1]*ax_pl_pos[3],
                                    ax_is_pos[2]*ax_pl_pos[2], ax_is_pos[3]*ax_pl_pos[3]])
                            
                            # plot all sample cross sections
                            for cs_ii, cs_l in enumerate(cs_l_tgb):
                                ax_ispl.plot(cs_l, cs_h_tgb[cs_ii], color=c_is_all,
                                        linewidth=lw_cs_all, zorder=101)
                            # plot mean cross section
                            ax_ispl.plot(cs_l_m_tgb, cs_h_m_tgb, c=c_surf,
                                    linewidth=lw_is, zorder=105)
                            # plot fitted triple trapezoid profile and vertices
                            ax_ispl.plot([-wsll_tgb, wslr_tgb], [hm_tgb, hm_tgb], c=c_wl,
                                    zorder=119, linewidth=lw_is)
                            ax_ispl.plot(int_l, ttp_fit_tgb, c=c_fit, ls='--',
                                    zorder=120, linewidth=lw_is)
                            ax_ispl.scatter(ttp_fit_p_tgb[:, 0], ttp_fit_p_tgb[:, 1],
                                       s=40 , marker='o', facecolors='none', linewidths=lw_is, 
                                       edgecolors=c_fit, zorder=150)

                            # set limits and labels of inset
                            ch_w = 40
                            ch_h = 3
                            ax_is.set_xlim(-ch_w/2, ch_w/2)
                            ax_is.set_ylim(-ch_h/4, ch_h)
                            ax_ispl.set_xlim(-ch_w/2, ch_w/2)
                            ax_ispl.set_ylim(-ch_h/4, ch_h)
                            xtcks = np.arange(-ch_w/2, ch_w/2+.1, ch_w/4, dtype=np.int)
                            ax_ispl.set_xticks(xtcks)
                            ytcks = np.arange(0, ch_h+.1, 1, dtype=np.int)
                            ax_ispl.set_yticks(ytcks)
                            ax_ispl.set_xticklabels(xtcks, fontsize=l_sz-4)
                            ax_ispl.set_yticklabels(ytcks, fontsize=l_sz-4)
                            ax_ispl.tick_params(axis='x', which='major', pad=2)
                            ax_ispl.tick_params(axis='y', which='major', pad=2)

                            # show figure, if activated
                            if ctrl_show_plots:
                                plt.show()
                            # save figure, if activated
                            if ctrl_save_plots:
                                # create folder if it does not exist
                                if not os.path.isdir(path_plots_out):
                                    os.mkdir(path_plots_out)
                                # save figure
                                plt.savefig('{0:s}ttp_est_tgb-{1:03d}.png'.format(
                                        path_plots_out, int(tgb)), dpi=300)
                            # close figure
                            plt.close(fig)
                        # fill log
                        ser_ttp_fit_log.at[tgb] = 'fit'
                        
                    # if estimated water surface level is empty, mark model element
                    # in log file
                    else:
                        ser_ttp_fit_log.at[tgb] = 'est'
        
            # print fitted trapezoid parameters
            if print_out: 
                pd.options.display.float_format = '{:,.1f}'.format
                print(pd.concat((ser_ttp_fit_log, df_ttp_fit, df_ttp_fit_r2), axis=1))
    
            return df_ttp_fit, ser_ttp_fit_log, df_ttp_fit_r2
        
        # %% calculation section
        # get array indices of recent and upstream routing elements using max. flow acc.
        if print_out: print('...get array indices of recent and upstream routing elements...')
        df_tgb_rout_rc, df_tgb_up_rout_rc = get_rout_arr_idx(
                ser_pp_row, ser_pp_col, ser_pp_fa, ser_tgb_up_nd, ser_tgb_type, str_routing)
        # get cross section digital elevation grid profiles
        if print_out: print('...get cross section digital elevation grid profiles...')
        ser_cs_l, ser_cs_h = get_cs_dem_prof(
                dem, fd, cellsz, gis_val_rows, gis_val_cols, gis_val_down_rows, gis_val_down_cols,
                df_tgb_rout_rc, df_tgb_up_rout_rc, ser_tgb_type, str_routing,
                def_cs_dist_eval=def_cs_dist_eval, def_cs_wmax_eval=def_cs_wmax_eval,
                print_out=print_out)
        # calculate mean cross section for each model element
        if print_out: print('...calculate mean cross section for each model element...')
        ser_cs_l_m, ser_cs_h_m = calc_mean_cs(
                ser_cs_l, ser_cs_h,
                def_cs_wmax_eval=def_cs_wmax_eval, def_flpl_wmax_eval=def_flpl_wmax_eval,
                def_ch_wmax_eval=def_ch_wmax_eval, 
                def_val_wres=def_val_wres, def_flpl_wres=def_flpl_wres,
                def_ch_w=def_ch_w, def_ch_h=def_ch_w, def_ch_wres=def_ch_wres)
        # estimate channel water surface level
        if print_out: print('...estimate channel water surface level...')
        h_ll, ser_wsll_ll, ser_wslm_ll, ser_wslr_ll, df_ch_h, _, _ = tc.est_ch_wsl(
                ser_cs_l_m, ser_cs_h_m, ser_tgb_q_in,
                def_cs_hmax_eval=def_cs_hmax_eval, def_lam_hres=def_lam_hres,
                def_ch_vmin=def_ch_vmin, def_ch_vmax=def_ch_vmax)
        # fit channel depth and width
        if print_out: print('...fit channel depth and width...')
        ser_ch_h_fit, df_ch_wsl_fit, _ = tc.fit_ch(
                ser_pef_hm, ser_pef_bm, ser_cs_l_m, ser_cs_h_m, 
                ser_wsll_ll, ser_wslm_ll, ser_wslr_ll, 
                def_cs_hmax_eval=def_cs_hmax_eval,
                def_ch_wmax_eval=def_ch_wmax_eval, def_lam_hres=def_lam_hres, 
                def_chbank_slmin=def_chbank_slmin, def_ch_hmin=def_ch_hmin,
                def_ch_hmin_eval=def_ch_hmin_eval, 
                ctrl_show_plots=ctrl_show_plots, ctrl_save_plots=ctrl_save_plots, 
                ser_tgb_a=ser_tgb_a, ser_tgb_a_in=ser_tgb_a_in, ser_tgb_q_in=ser_tgb_q_in, 
                def_ch_hmax_eval=def_ch_hmax_eval, path_plots_out=path_plots_out)
        # fit foreland and valley
        if print_out: print('...fit foreland and valley...')
        ser_ch_wsll_fit = df_ch_wsl_fit.left
        ser_ch_wslr_fit = df_ch_wsl_fit.right
        df_ttp_fit, ser_ttp_fit_log, df_ttp_fit_r2 = fit_ttp(
                ser_cs_l_m, ser_cs_h_m,
                ser_ch_wsll_fit, ser_ch_wslr_fit, ser_ch_h_fit, 
                def_cs_hmax_eval=def_cs_hmax_eval, print_out=print_out,
                ctrl_show_plots=ctrl_show_plots, ctrl_save_plots=ctrl_save_plots,
                ser_cs_l=ser_cs_l, ser_cs_h=ser_cs_h,
                ser_tgb_a=ser_tgb_a, ser_tgb_a_in=ser_tgb_a_in, ser_tgb_q_in=ser_tgb_q_in,
                def_cs_wmax_eval=def_cs_wmax_eval, path_plots_out=path_plots_out)
        
        return df_ttp_fit, ser_ttp_fit_log, df_ttp_fit_r2
    
    # %% calculations
    # internal variable field names
    f_pp       = 'pp'
    f_pp_down  = 'pp_down'
    f_fa       = 'fa'
    f_fl       = 'fl'
    f_row      = 'row'
    f_col      = 'col'
    f_tgb      = 'tgb'
    f_tgb_down = 'tgb_down'
    f_tgb_type = 'tgb_type'
    f_tgb_dtgb = 'tgb_dtgb'
    f_tgb_tgbt = 'tgb_tgbt'
    f_x        = 'x'
    f_y        = 'y'
    f_ain      = 'a_in'
    f_qin      = 'q_in'
    f_acum     = 'a_cum'
    f_hu       = 'hu'
    f_l        = 'l'
    f_hot      = 'hot'
    f_hut      = 'hut'
    f_tal      = 'tal'
    f_ft       = 'ft'
    f_nrflv    = 'nrflv'
    f_chzmin   = 'ch_z_min'
    f_chzmax   = 'ch_z_max'
    f_hm       = 'hm'
    f_bm       = 'bm'
    # model element types and default values
    str_headw    = 'headwater'
    str_routing  = 'routing'
    str_dummy    = 'dummy'
    val_no_dummy = -1
    # define intermediate paths
    name_pp_sc_corr = 'pp_sc_corr'
    name_tgb_down_p = 'tgb_down_p'
    # arcpy field names (ONLY CHANGE IF ARCPY FUNCTION CHANGES FORCE YOU!)
    f_p_x      = 'POINT_X'
    f_p_y      = 'POINT_Y'
    f_shp_a    = 'Shape_Area'
    
    # read gis data
    if print_out: print('...process flow direction, accumulation, and length...')
    os.chdir(path_gdb_out)
    # import elevation, flow direction, accumulation and length as numpy rasters
    dem, ncols, nrows, cellsz, xll, yll, ctrl_tif_export = tc.fdal_raster_to_numpy(
            path_dem_c, 'dem', path_files_out, True)
    fd, _, _, _, _, _, _ = tc.fdal_raster_to_numpy(
            path_fd_c,  'fd',  path_files_out,  True)
    fa, _, _, _, _, _, _ = tc.fdal_raster_to_numpy(
            path_fa_c,  'fa',  path_files_out,  True)
    fl, _, _, _, _, _, _ = tc.fdal_raster_to_numpy(
            path_fl_c,  'fl',  path_files_out,  True)
    # add a NaN boundary to all gis input data sets
    empty_row = np.zeros((1,     ncols)) * np.nan
    empty_col = np.zeros((nrows + 2, 1)) * np.nan
    dem = np.concatenate((empty_row, dem, empty_row), axis=0)
    dem = np.concatenate((empty_col, dem, empty_col), axis=1)
    fd  = np.concatenate((empty_row, fd, empty_row), axis=0)
    fd  = np.concatenate((empty_col, fd, empty_col), axis=1)
    fa  = np.concatenate((empty_row, fa, empty_row), axis=0)
    fa  = np.concatenate((empty_col, fa, empty_col), axis=1)
    fl  = np.concatenate((empty_row, fl, empty_row), axis=0)
    fl  = np.concatenate((empty_col, fl, empty_col), axis=1)
    # adjust gis parameters for new sizes
    ncols += 2
    nrows += 2
    xll   -= cellsz
    yll   -= cellsz
    # set flow accumulation thresholds
    fa_thr_sc_area = def_sc_area / cellsz
    fa_min_thr_sc_area = fa_thr_sc_area * 0.5
    # set arrays to nan if threshold is not met
    with np.errstate(invalid='ignore'):
        fa_l_thr_ii = fa < fa_min_thr_sc_area
    fd[fa_l_thr_ii] = np.nan
    fl[fa_l_thr_ii] = np.nan
    # Analyze GIS data for not-NaNs
    gis_val_ii   = np.nonzero(~np.isnan(fd))
    gis_val_cols = gis_val_ii[1]
    gis_val_rows = gis_val_ii[0]
    gis_val_ct   = gis_val_rows.shape[0]
    # prepare flow direction look up table for x and y cell number differences
    fd_lu = pd.DataFrame(
                np.array([[ 1, 0], [ 1, 1], [ 0, 1], [-1, 1],
                          [-1, 0], [-1,-1], [ 0,-1], [ 1,-1]]),
                index=[1, 2, 4, 8, 16, 32, 64, 128], columns=['dx', 'dy'])
    # pre-allocate arrays
    if ncols * nrows <= 32767: np_type = np.int32
    else: np_type = np.int64
    fd_xd = np.empty((gis_val_ct, 1), dtype=np_type)
    fd_yd = np.empty((gis_val_ct, 1), dtype=np_type)
    # iterate flow direction integer values
    for fdir, (dx, dy) in fd_lu.iterrows():
        fd_notnans_ii = fd[~np.isnan(fd)] == fdir
        fd_xd[fd_notnans_ii] = dx
        fd_yd[fd_notnans_ii] = dy
    # calculate downstream cell indices (column and row)
    gis_val_down_cols = gis_val_cols + np.int64(fd_xd[:, 0])
    gis_val_down_rows = gis_val_rows + np.int64(fd_yd[:, 0])
    
    # import corrected pour point feature class
    if print_out: print('...import corrected pour point feature class...')
    # add X- and Y-coordinates to corrected pour point feature class
    path_pp_sc_corr = path_gdb_out + name_pp_sc_corr
    arcpy.management.CopyFeatures(path_pp_sc, path_pp_sc_corr, '', None, None, None)
    arcpy.management.AddXY(path_pp_sc_corr)
    # import corrected pour points as numpy array and convert to pandas DataFrame
    import_fields = (f_pp, f_p_x, f_p_y)
    np_pp_sc = arcpy.da.FeatureClassToNumPyArray(path_pp_sc_corr, import_fields)
    pp_sc_id = np_pp_sc[f_pp].astype(np.int)
    df_pp_sc = pd.DataFrame(np_pp_sc[[f_p_x, f_p_y]], index=pp_sc_id)
    df_pp_sc = df_pp_sc.rename(columns={f_p_x: f_x, f_p_y: f_y})
    # check all coordinates to be inside the dem raster domain
    pp_sc_in_dem_x_bool = np.logical_and(
            df_pp_sc[f_x] >= xll,
            df_pp_sc[f_x] <= xll + ncols + cellsz)
    pp_sc_in_dem_y_bool = np.logical_and(
            df_pp_sc[f_y] >= yll,
            df_pp_sc[f_y] <= yll + nrows + cellsz)
    # if one is outside, throw error message
    if not np.logical_and(np.all(pp_sc_in_dem_x_bool), np.all(pp_sc_in_dem_y_bool)):
        pp_sc_outside_dem = df_pp_sc[np.any((~pp_sc_in_dem_x_bool,
                                             ~pp_sc_in_dem_y_bool), 0)]
        print('ERROR: The following pour points are outside of elevation raster domain:')
        print(pp_sc_outside_dem)
        sys.exit()
    
    # get raster coordinates, flow accumulation and length for pour points
    if print_out: print('...get raster row and column, flow accumulation and length...')
    # get raster coordinates
    arr_x =           np.linspace(xll + 0.5, xll + ncols - 0.5, ncols)
    arr_y = np.flipud(np.linspace(yll + 0.5, yll + nrows - 0.5, nrows))
    # calculate pour point indices in array
    pp_row = []
    pp_col = []
    for pp in df_pp_sc.index:
        pp_row.append(np.argmin(np.abs(arr_y - df_pp_sc.loc[pp, f_y])))
        pp_col.append(np.argmin(np.abs(arr_x - df_pp_sc.loc[pp, f_x])))
    df_pp_sc.at[:, f_row] = pp_row
    df_pp_sc.at[:, f_col] = pp_col
    # get elevation, flow accumulation and length values
    df_pp_sc.at[:, f_hu] = dem[df_pp_sc.loc[:, f_row], df_pp_sc.loc[:, f_col]]
    df_pp_sc.at[:, f_fa] =  fa[df_pp_sc.loc[:, f_row], df_pp_sc.loc[:, f_col]].astype(np.int)
    df_pp_sc.at[:, f_fl] =  fl[df_pp_sc.loc[:, f_row], df_pp_sc.loc[:, f_col]]
    # if one is outside, throw error message
    if np.any(np.isnan(df_pp_sc.fa)):
        pp_sc_outside_fa = df_pp_sc[np.isnan(df_pp_sc.fa)]
        print('ERROR: The following pour points are outsode of flow accumulation domain:')
        print(pp_sc_outside_fa.loc[:, [f_x, f_y]])
        sys.exit()
    else:
        df_pp_sc = df_pp_sc.astype({f_fa: 'int32'})
    
    # get relations of pour points
    if print_out: print('...get relations of pour points...')
    # establish downstream relation of pour points
    df_pp_sc[f_pp_down] = get_downstream_pp(copy.deepcopy(df_pp_sc.loc[:, f_fl]),
            df_pp_sc.loc[:, f_row], df_pp_sc.loc[:, f_col],
            fd, gis_val_rows, gis_val_cols, gis_val_down_rows, gis_val_down_cols,
            print_out=print_out)
    # establish upstream relation of pour points
    df_pp_up = get_upstream_pp(df_pp_sc.loc[:, f_pp_down])
    # calculate final model network
    # define key-words to identify element types
    df_tgb, df_tgb_up = calc_model_nw(df_pp_sc, df_pp_up,
                           str_headw, str_routing, str_dummy, val_no_dummy,
                           print_out=print_out)
    # get up- and downstream model cell indices while ignoring dummy elements
    ser_tgb_up_nd = tc.get_upstream_idx_ign_dumm(
            df_tgb.loc[:, f_tgb_down], df_tgb.loc[:, f_tgb_type] == str_headw,
            df_tgb.loc[:, f_tgb_type] == str_dummy)
    # transform pour point information to model subcatchment information
    # get index of pour points ignoring dummies
    j_idx = np.array(df_tgb.loc[:, f_tgb_tgbt])
    sel_bool = df_tgb.loc[:, f_tgb_type] == str_dummy
    j_idx[sel_bool] = df_tgb.loc[j_idx[sel_bool], f_tgb_dtgb]
    # calculate information for all catchments
    df_tgb.loc[:, f_acum] = df_pp_sc.loc[j_idx, f_fa].values * ((cellsz / 1000) ** 2)
    df_tgb.loc[:,   f_fl] = df_pp_sc.loc[j_idx, f_fl].values * cellsz
    df_tgb.loc[:,    f_x] = df_pp_sc.loc[j_idx,  f_x].values
    df_tgb.loc[:,    f_y] = df_pp_sc.loc[j_idx,  f_y].values
    df_tgb.loc[:,   f_hu] = df_pp_sc.loc[j_idx, f_hu].values
    # set special values for headwater and routing subcatchments
    sel_bool = np.isin(df_tgb.loc[:, f_tgb_type], [str_routing, str_headw])
    df_tgb.loc[sel_bool, f_row] = df_pp_sc.loc[
            df_tgb.loc[sel_bool, f_tgb_tgbt], f_row].values
    df_tgb.loc[sel_bool, f_col] = df_pp_sc.loc[
            df_tgb.loc[sel_bool, f_tgb_tgbt], f_col].values
    # calculate geo-information for channel routing
    df_tgb_ch_gis = calc_ch_gis(ser_tgb_up_nd, df_tgb.loc[:, f_tgb_type],
                                df_tgb.loc[:, f_acum], df_tgb.loc[:, f_fl],
                                df_tgb.loc[:, f_hu], str_headw, str_routing)
    
    # calculate minimum model subcatchment elevation for runoff concentration
    # estimation (hut)
    if print_out: print('...calculate minimum elevation for runoff concentration...')
    # define HUT for head waters as low point of cell
    ser_hut = df_tgb_ch_gis.loc[:, f_chzmin] \
        + (df_tgb_ch_gis.loc[:, f_chzmax] \
           - df_tgb_ch_gis.loc[:, f_chzmin]) * def_zmin_rout_fac
    ser_hut.at[df_tgb.loc[:, f_tgb_type] == str_headw] \
        = df_tgb_ch_gis.loc[df_tgb.loc[:, f_tgb_type] == str_headw, f_chzmin]
    ser_hut.name = f_hut
    
    # calculate routing parameters
    if print_out: print('...calculate routing parameters...')
    # calculate channel slope
    ser_ch_zdif = df_tgb_ch_gis.loc[:, f_chzmax] - df_tgb_ch_gis.loc[:, f_chzmin]
    ser_ch_sl = tc.calc_ch_sl(ser_ch_zdif, df_tgb_ch_gis.loc[:, f_l],
                              df_tgb.loc[:, f_tgb_type] == str_routing,
                              def_sl_excl_quant=def_sl_excl_quant)
    # calculate final routing flow lengths for model elements
    df_tgb_rout_fl = calc_ch_rout_fl(
            df_tgb.loc[:, f_tgb_type], df_tgb.loc[:, f_tgb_down],
            df_tgb_ch_gis.loc[:, f_l], str_headw, str_routing, str_dummy)
    
    # export calculated parameters to point feature class
    if print_out: print('...export calculated parameters to point feature class...')
    # summarize information for export
    ser_tgb = df_tgb.index.to_series(name=f_tgb)
    df_data_tgb_p = pd.concat([ser_tgb,
                               df_tgb.loc[:, [f_tgb_down, f_tgb_type, f_tgb_dtgb, f_x, f_y]],
                               df_tgb_rout_fl], axis=1)
    # get spatial reference
    sr_obj = arcpy.Describe(path_fd_c).spatialReference
    # export to point feature classes
    tc.tgb_to_points(df_data_tgb_p, sr_obj, path_gdb_out, name_tgb_p,
                     geometry_fields=(f_x, f_y))
    tc.tgb_to_points(df_data_tgb_p, sr_obj, path_gdb_out, name_tgb_down_p,
                     geometry_fields=(f_x, f_y))
    # calculate and correct sub-watershed polygons
    if print_out: print('...calculate and correct subcatchment polygons...')
    path_tgb_p = path_gdb_out + name_tgb_p
    calc_corr_sc_polyg(path_fd_c, path_tgb_p, val_no_dummy, def_a_min_tol, 
                       path_gdb_out, name_tgb_sj=name_tgb_sj)
    
    # calculate subcatchment runoff concentration parameters
    if print_out: print('...calculate subcatchment runoff concentration parameters...')
    path_tgb_sj = path_gdb_out + name_tgb_sj
    calc_sc_roconc_pars(path_dem_c, path_fl_c, path_tgb_p, path_tgb_sj,
                        path_gdb_out, str_headw, str_routing)
    
    # calculate coordinates
    if print_out: print('...calculate coordinates...')
    calc_sc_coords(path_tgb_p, path_tgb_sj, path_gdb_out)
    
    # summarize information to data frame
    if print_out: print('...summarize information...')
    # import point feature class
    fields = [f_tgb, f_x, f_y, f_hot, f_tal]
    np_tgb_p = arcpy.da.FeatureClassToNumPyArray(path_tgb_p, fields)
    df_tgb_p = pd.DataFrame(np_tgb_p, index=np_tgb_p[f_tgb])
    df_roconc_params = pd.concat((ser_hut, df_tgb_p.loc[:, [f_hot, f_tal]]), axis=1)
    # calculate final subcatchment area
    np_tgb_s = arcpy.da.FeatureClassToNumPyArray(path_tgb_sj, [f_tgb, f_shp_a])
    df_tgb.loc[:, f_ft] = pd.Series(np_tgb_s[f_shp_a],
              index=np_tgb_s[f_tgb], name=f_ft) / 10**6
    # calculate inflow catchment size informative value
    df_tgb.loc[:, f_ain] = df_tgb.loc[:, f_acum] - df_tgb.loc[:, f_ft]
    df_tgb.loc[df_tgb.loc[:, f_tgb_type] != str_routing, f_ain] = 0
    
    # calculate cross section parameters
    # calculate specific discharge
    q_spec = hq_ch / hq_ch_a
    # calculate characteristic river-forming discharge
    df_tgb.loc[:, f_qin] = tc.calc_ch_form_q(
            df_tgb.loc[:, f_ain], df_tgb.loc[:, f_tgb_down],
            q_spec, ser_q_in_corr=ser_q_in_corr)
    # calculate tripel trapezoid river cross sections using estimator function
    df_ttp = tc.calc_ttp(df_tgb.loc[:, f_qin],
                         df_tgb.loc[:, f_tgb_type] == str_routing,
                         ch_est_method=ch_est_method, def_bx=def_bx,
                         def_bbx_fac=def_bbx_fac, def_bnm=def_bnm, 
                         def_bnx=def_bnx, def_bnvrx=def_bnvrx, 
                         def_skm=def_skm, def_skx=def_skx,
                         print_out=print_out)
    # use fit tripel trapezoid river cross sections if required
    if ctrl_rout_cs_fit:
        if print_out: print('...tripel trapezoid river cross section fitting activated...')
        # fit tripel trapezoid to the mean of multiple derived cross sections
        df_ttp_fit, ser_ttp_fit_log, df_ttp_fit_r2 = fit_mean_ttp(
                df_ttp.loc[:, f_hm], df_ttp.loc[:, f_bm],
                df_pp_sc.loc[:, f_row], df_pp_sc.loc[:, f_col],
                df_pp_sc.loc[:, f_fa], df_tgb.loc[:, f_qin],
                dem, fd, cellsz,
                gis_val_rows, gis_val_cols, gis_val_down_rows, gis_val_down_cols,
                ser_tgb_up_nd, df_tgb.loc[:, f_tgb_type], str_routing,
                def_cs_dist_eval=def_cs_dist_eval, 
                def_cs_wmax_eval=def_cs_wmax_eval, def_cs_hmax_eval=def_cs_hmax_eval,
                def_flpl_wmax_eval=def_flpl_wmax_eval,
                def_ch_wmax_eval=def_ch_wmax_eval, def_ch_hmin_eval=def_ch_hmin_eval,
                def_ch_hmax_eval=def_ch_hmax_eval,
                def_ch_hmin=def_ch_hmin, def_ch_vmin=def_ch_vmin,
                def_ch_vmax=def_ch_vmax, def_chbank_slmin=def_chbank_slmin,
                def_val_wres=def_val_wres, def_flpl_wres=def_flpl_wres,
                def_ch_w=def_ch_w, def_ch_h=def_ch_h, 
                def_ch_wres=def_ch_wres, def_lam_hres=def_lam_hres, 
                print_out=print_out,
                ctrl_show_plots=ctrl_show_plots, ctrl_save_plots=ctrl_save_plots,
                ser_tgb_a=df_tgb.loc[:, f_ft], ser_tgb_a_in=df_tgb.loc[:, f_ain], 
                path_plots_out=path_plots_out)
        # merge estimated and fit parameters
        df_ttp_fit = df_ttp_fit.loc[ser_ttp_fit_log == 'fit', :]
        df_ttp.at[df_ttp_fit.index, df_ttp_fit.columns] = df_ttp_fit
    
    # summarize Series to DataFrame
    fields = ['TGB','NRFLV','FT','HUT','HOT','TAL','X','Y','KMU','KMO','GEF',
        'HM','BM','BL','BR','BBL','BBR','BNM','BNL','BNR','BNVRL','BNVRR',
        'SKM','SKL','SKR','Kommentar_EZG-A','Kommentar_GBA']
    df_data_tgbdat = pd.concat([df_tgb_p.loc[:, f_tgb], df_tgb.loc[:, f_nrflv],
                                np.round(df_tgb.loc[:, f_ft], 3), df_roconc_params, 
                                df_tgb_p.loc[:, [f_x, f_y]].astype(np.int),
                                df_tgb_rout_fl, ser_ch_sl, df_ttp, df_tgb.loc[:, f_ain],
                                df_tgb.loc[:, f_qin]], axis=1)
    df_data_tgbdat.columns=fields
    # correct data for headwater and dummy catchments
    df_data_tgbdat.at[df_tgb.loc[:, f_tgb_type] == str_headw,
                      ['GEF', 'HM','BM','BL','BR','BBL','BBR','BNM',
                       'BNL','BNR','BNVRL','BNVRR','SKM','SKL','SKR']] = np.nan
    df_data_tgbdat.at[df_tgb.loc[:, f_tgb_type] == str_dummy,
                      ['HUT','HOT','TAL','GEF','HM','BM','BL','BR','BBL','BBR',
                       'BNM','BNL','BNR','BNVRL','BNVRR','SKM','SKL','SKR']] = np.nan
    
    return df_data_tgbdat