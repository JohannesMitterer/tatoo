# -*- coding: utf-8 -*-
"""
This library contains all functions of the TATOO core library, which are 
referenced by the libraries 'TATOO raster' and 'TATOO subcatchment'.

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
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy.interpolate import griddata
from scipy.integrate import cumtrapz
from scipy.signal import find_peaks


# %% function to create a pour point feature class with user input variables
def create_pourpoint(path_fnw,
                     path_gdb_out, name_pp='pp', field_pp='ModelWatershed',
                     print_out=False):
    """
    Creates a point feature class in the defined file geodatabase to be filled
    by the user with pour points. The point feature class has neither Z- nor M-values.
    The attribute table of the feature class includes a 'short' variable 'ModelWatershed'.

    JM 2021

    Arguments:
    -----------
    path_fnw: str
        path of the flow network feature class or shape file (e.g., 'c:\fnw.shp')
    path_gdb_out: str
        path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
    name_pp: str
        name of the output pour point feature class (e.g., 'pp')
    field_pp_ws: str
        name of the field in path_pp_ws containing the resulting watershed ID numbers
        and negative numbers for watersheds to be excluded
        (e.g., 'ModelWatershed')
    print_out: boolean (optional, default: False)
        true if workprogress shall be print to command line

    Returns:
    -----------
    Saves the output pour point feature class
    
    """
    
    if print_out: print('...create pour point feature class...')
    # prepare point feature class for model pour points
    path_pp = path_gdb_out + name_pp
    if arcpy.Exists(path_pp):
        arcpy.management.Delete(path_pp)
    sr = arcpy.Describe(path_fnw).spatialReference
    arcpy.CreateFeatureclass_management(path_gdb_out, name_pp, 'POINT', '',
                                        'DISABLED', 'DISABLED', sr, '', '0',
                                        '0', '0', '')
    # create field for user input
    path_pp = path_gdb_out + '\\' + name_pp
    arcpy.AddField_management(path_pp, field_pp,
                              'SHORT', '', '', '', '', 'NULLABLE', 'NON_REQUIRED', '')
    arcpy.CalculateField_management(path_pp, field_pp, '1', 'PYTHON3')

# %% import elevation, flow direction, accumulation and length as numpy rasters
def fdal_raster_to_numpy(path_in_raster, raster_type,
                         path_out_tif, ctrl_del=False):
    """
    This function imports elevation, flow direction, flow accumulation, and 
    flow length raster files to numpy arrays and returns some characteristics:
    number of clumns, number of rows, cellsize, and x- and y-coordinates
    of the lower left corner of the raster. As the ESRI ArcGIS function
    RasterToNumPyArray() allows rasters until a system specific block size,
    this function utilizes the GDAL library instead if necessary. The user
    may additionally define the raster type choosing from 'dem', 'fd' (flow
    direction), 'fa' (flow accumulation), and 'fl' (flow length) to ensure
    proper no-data and data type import. Finally, the user may define wether
    the potentially necessary GDAL import via TIFF conversion shall leave the
    TIFF file on the hard drive or not.

    JM 2021

    Arguments:
    -----------
    path_in_raster: str
        path of the input ESRI ArcGIS raster (e.g., 'c:\model_creation.gdb\dem')
    raster_type: str
        string defining the type of the input raster. The user may choose out of
            - 'dem': digital elevation model raster (float conversion)
            - 'fd': flow direction raster (float conversion and no-data handling)
            - 'fa': flow accumulation raster (standard)
            - 'fl': flow length raster (standard)
    path_out_tif: str
        path where temporary files may be stored if GDAL import is necessary
        (e.g., 'c:\tmp_model_files\')
    ctrl_del: boolean (optional, default: False)
        swith to control if the created TIFF is left on the hard drive (False) or 
        it is deleted (True)

    Returns:
    -----------
    fdal: numpy.Array
        numpy array containing the input ArcGIS Raster
    ncols: int
        number of columns of the imported raster
    nrows: int
        number of rows of the imported raster
    cellsz: float
        cell size of the imported raster
    xll: float
        x-coordinate of the lower left corner of the imported raster
    yll: float
        y-coordinate of the lower left corner of the imported raster
    ctrl_tif_export: boolean
        boolean indicating GDAL import (True, TIFF conversion) or arcpy 
        import (False) otherwise.
    """
    # use arcpy functions
    try:
        # get raster handle and import to numpy array
        gis_fdal = arcpy.Raster(path_in_raster)
        fdal = arcpy.RasterToNumPyArray(gis_fdal, nodata_to_value = -1)
        # set data type and handle no-data
        if raster_type in ['dem', 'fd']: fdal = fdal.astype(float)
        fdal[fdal==-1] = np.nan
        # get raster properties
        ncols = gis_fdal.width
        nrows = gis_fdal.height
        cellsz = gis_fdal.meanCellHeight
        xll = gis_fdal.extent.XMin
        yll = gis_fdal.extent.YMin
        # set switch for export
        ctrl_tif_export = False
    # if raster is too large for arcpy function, import with GDAL
    except:
        # convert raster to TIFF file
        if arcpy.Exists(path_out_tif):
            arcpy.Delete_management(path_out_tif)
        arcpy.management.CopyRaster(path_in_raster, path_out_tif, '', None,
                                    '3,4e+38', 'NONE', 'NONE', '', 'NONE',
                                    'NONE', 'TIFF', 'NONE',
                                    'CURRENT_SLICE', 'NO_TRANSPOSE')
        # get raster handle and import to numpy array using GDAL
        fdal_fid = gdal.Open(path_out_tif)
        fdal = fdal_fid.ReadAsArray()
        # get and convert no-data value to np.nan and handle data type
        srcband = fdal_fid.GetRasterBand(1)
        nodata_val = srcband.GetNoDataValue()
        if raster_type in ['dem', 'fd']: 
            fdal = fdal.astype(float)
        if raster_type == 'fd':
            fdal[~np.isin(fdal, [1, 2, 4, 8, 16, 32, 64, 128])] = np.nan
        else:
            fdal[fdal == nodata_val] = np.nan
        # delete temporary TIF file if required
        if ctrl_del: arcpy.management.Delete(path_out_tif)
        # get raster properties
        ncols = fdal_fid.RasterXSize
        nrows = fdal_fid.RasterYSize
        ulx, cellsz, xskew, uly, yskew, yres = fdal_fid.GetGeoTransform()
        xll = ulx
        yll = uly + (fdal_fid.RasterYSize * yres)
        # clear handles
        srcband = None
        fdal_fid = None
        # set switch for export
        ctrl_tif_export = True
        # return
    return fdal, ncols, nrows, cellsz, xll, yll, ctrl_tif_export

# %% export numpy array to ArcGIS Raster using a pre-defined format
def numpy_to_predef_gtiff_fmt(
        np_array_in, xll, yll, cellsz, path_gdb_out, name_raster_out,
        ctrl_tif_export=False, path_gdb_fmt_in='', name_raster_fmt_in='',
        path_folder_tif_tmp=''):
    """
    This function exports a numpy array to an ArcGIS Raster.
    As the ESRI ArcGIS function NumPyArrayToRaster() only allows rasters
    until a system specific block size, this function utilizes the GDAL
    library instead if necessary (ctrl_tif_export = True) using a 
    pre-defined raster format.

    JM 2021

    Arguments:
    -----------
    np_array_in: numpy.Array
        numpy array containing the input to be converted into ArcGIS Raster
    xll: float
        x-coordinate of the lower left corner of the exported raster [m]
    yll: float
        y-coordinate of the lower left corner of the exported raster [m]
    cellsz: float
        cell size of the imported raster [m] (e.g., 100)
    path_gdb_out: str
        path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
    name_raster_out: str
        name of the output ESRI ArcGIS raster (e.g., 'dem')
    ctrl_tif_export: boolean (optional, default: False)
        boolean indicating GDAL export (True, TIFF conversion) or arcpy 
        import (False) otherwise
    path_gdb_fmt_in: str (optional, default: '')
        path of the file geodatabase of the format blueprint raster file
        (e.g., 'c:\model_creation.gdb')
    name_raster_fmt_in: str (optional, default: '')
        name of the format blueprint raster file
        (e.g., 'dem')
    path_folder_tif_tmp: str (optional, default: '')
        path where temporary files may be stored if GDAL import is necessary
        (e.g., 'c:\tmp_model_files\')

    Returns:
    -----------
    This function saves the resulting raster file in the defined file geodatabase.
    """        
    # define path of output raster
    path_out_raster = path_gdb_out + name_raster_out
    # export using arcpy library
    if not ctrl_tif_export:
        # create raster
        fa_dif_r = arcpy.NumPyArrayToRaster(
                np_array_in, arcpy.Point(xll, yll), cellsz, cellsz, np.nan, None)
        # export raster to geodatabase
        fa_dif_r.save(path_out_raster)
    else:
        # define temporary paths
        path_in_raster_fmt = path_gdb_fmt_in + name_raster_fmt_in
        path_tmp_tif_fmt = path_folder_tif_tmp + name_raster_fmt_in + '.tif'
        path_tmp_tif = path_folder_tif_tmp + name_raster_out + '.tif'
        # convert raster to TIFF file
        if arcpy.Exists(path_tmp_tif_fmt):
            arcpy.Delete_management(path_tmp_tif_fmt)
        arcpy.management.CopyRaster(
                path_in_raster_fmt, path_tmp_tif_fmt, '', None, '3,4e+38', 'NONE', 'NONE',
                '', 'NONE', 'NONE', 'TIFF', 'NONE', 'CURRENT_SLICE', 'NO_TRANSPOSE')
        # open original GeoTIFF as object
        dem_o = gdal.Open(path_tmp_tif_fmt)
        # open raster band as object
        srcband = dem_o.GetRasterBand(1)
        # load data of raster band as numpy array
        dem = srcband.ReadAsArray()
        # get numbers of rows and cols of numpy array
        [cols, rows] = dem.shape
        # load driver for GeoTIFF format
        driver = gdal.GetDriverByName('GTiff')
        # create writable file with same size as input raster and dtype=float32
        outdata = driver.Create(path_tmp_tif, rows, cols, 1, gdal.GDT_Float32)
        # set geotransform attribute same as input
        outdata.SetGeoTransform(dem_o.GetGeoTransform())
        # set projection same as input
        outdata.SetProjection(dem_o.GetProjection())
        # write array to raster band
        outdata.GetRasterBand(1).WriteArray(np_array_in[1:-1,1:-1])
        # set same nodata value as input
        outdata.GetRasterBand(1).SetNoDataValue(srcband.GetNoDataValue())
        # save created raster to disk
        outdata.FlushCache()
        # release handles
        outdata = None
        srcband = None
        dem_o = None
        # convert TIFF to ArcGIS Raster
        arcpy.management.CopyRaster(
                path_tmp_tif, path_out_raster, '', None, '3,4e+38', 'NONE', 'NONE',
                '32_BIT_FLOAT', 'NONE', 'NONE', 'GRID', 'NONE', 'CURRENT_SLICE',
                'NO_TRANSPOSE')
        # delete GeoTIFF files
        arcpy.management.Delete(path_tmp_tif_fmt)
        arcpy.management.Delete(path_tmp_tif)

# %% function to find all upstrem model cell indices
def get_upstream_idx(ser_tgb_down):
    """
    This function finds all upstream model elements using the index and the
    downstream relation.

    JM 2021

    Arguments:
    -----------
    ser_tgb_down: pandas.Series
        Series of downstream model element indices corresponding to the serie's
        ascending index. The last value is outlet, identified with a zero.
        The outlet will be neglected in calculations.
        (e.g., pd.Series([2, 4, 4, 0], index=[1, 2, 3, 4], name='tgb_down'))

    Returns:
    -----------
    ser_tgb_up: pandas.Series
        Series of corresponding upstream model element indices.
        Dummy elements are represented as empty array (e.g., []).
    """
    # pre-allocate list of upstream indices
    ser_tgb_up = pd.Series(ser_tgb_down.shape[0]*[[]], index=ser_tgb_down.index,
                     name='tgb_up')
    # iterate downstream index array
    for tgb, tgb_down in ser_tgb_down.iteritems():
        # if the model element is not the outlet, calculate upstream
        # elements' indices
        if tgb_down != 0:
            # find position in list to add the found upstream index
            ser_tgb_up.at[tgb_down] = np.array(
                    np.hstack((ser_tgb_up.at[tgb_down], tgb)), dtype=np.int)
    # return upstream index list
    return ser_tgb_up

# %% function to find all downstream model cell indices while ignoring dummy cells
def get_downstream_idx_ign_dumm(ser_tgb_down, ser_tgb_type_dummy):
    """
    This function finds all downstream model elements using the index and the
    downstream relation and it ignores dummy elements.

    JM 2021

    Arguments:
    -----------
    ser_tgb_down: pandas.Series
        Series of downstream model element indices corresponding to the serie's
        ascending index. The last value is outlet, identified with a zero.
        The outlet will be neglected in calculations.
        (e.g., pd.Series([2, 4, 4, 0], index=[1, 2, 3, 4], name='tgb_down'))
    ser_tgb_type_dummy: pandas.Series
        Boolean Series, which identifies the dummy cells corresponding to the
        serie's ascending index with True.
        (e.g., pd.Series(data=[0, 0, 0, 0],
                         index=[1, 2, 3, 4], name='dummy', dtype='bool'))

    Returns:
    -----------
    ser_tgb_down_nd: pandas.Series
        Series of corresponding downstream model element indices ignoring dummy
        elements. Model outlet remains -1 and dummy elements are represented as 0.
    """
    # preallocate no-dummy index arrays
    ser_tgb_down_nd = pd.Series(np.zeros(ser_tgb_down.shape),
                              index=ser_tgb_down.index,
                              name='tgb_down_nd', dtype=np.int64)
    # Iterate over all index values unless outlet to find cell index of next real
    # downstream cell (jumping over dummy cell)
    for tgb, tgb_down in ser_tgb_down[:-1].iteritems():
        # if cell is no dummy get downstream index
        if not ser_tgb_type_dummy.at[tgb]:
            # get downstream index and iterate as long as downstream cell
            # is a dummy cell
            while ser_tgb_type_dummy.at[tgb_down]:
                tgb_down = ser_tgb_down.at[tgb_down]
            # if downstream cell is no dummy write downstream tgb to J_nd
            ser_tgb_down_nd.at[tgb] = tgb_down
        # otherwise set index to zero
        else:
            ser_tgb_down_nd.at[tgb] = -1
    return ser_tgb_down_nd

# %% function to find all upstream model cell indices while ignoring dummy cells
def get_upstream_idx_ign_dumm(ser_tgb_down, ser_tgb_type_headw, ser_tgb_type_dummy):
    """
    This function finds all downstream model elements using the index and the
    downstream relation and it ignores dummy elements.

    JM 2021

    Arguments:
    -----------
    ser_tgb_down: pandas.Series
        Series of downstream model element indices corresponding to the serie's
        ascending index. The last value is outlet, identified with a zero.
        The outlet will be neglected in calculations.
        (e.g., pd.Series([2, 4, 4, 0], index=[1, 2, 3, 4], name='tgb_down'))
    ser_tgb_type_headw: pandas.Series
        Boolean Series, which identifies the headwater cells corresponding to the
        serie's ascending index with True.
        (e.g., pd.Series(data=[1, 1, 0, 0],
                         index=[1, 2, 3, 4], name='headwater', dtype='bool'))
    ser_tgb_type_dummy: pandas.Series
        Boolean Series, which identifies the dummy cells corresponding to the
        serie's ascending index with True.
        (e.g., pd.Series(data=[0, 0, 0, 0],
                         index=[1, 2, 3, 4], name='dummy', dtype='bool'))

    Returns:
    -----------
    ser_tgb_up_nd: pandas.Series
        Series of corresponding upstream model element indices ignoring dummy
        elements. These are represented as empty array (e.g., []).
    """
    # pre-allocate list of upstream indices
    ser_tgb_up_nd = pd.Series(ser_tgb_down.shape[0]*[[]],
                            index=ser_tgb_down.index, name='tgb_up_nd')
    # Iterate over all index values unless outlet to find cell index of next real
    # upstream cell (jumping over dummy cell)
    for tgb, tgb_down in ser_tgb_down.iteritems():
        # if cell is no headwater find tgb of all real upstream cells
        # leaving dummy cells
        if not ser_tgb_type_headw.at[tgb]:
            # get upstream model elements, which are not of dummy type
            tgb_up = ser_tgb_down.index.values[ser_tgb_down == tgb]
            tgb_up_nd = tgb_up[~ser_tgb_type_dummy.loc[tgb_up]]
            # if cell is no dummy add upstream index
            if not ser_tgb_type_dummy.at[tgb]:
                ser_tgb_up_nd.at[tgb] = np.array(
                        np.concatenate((ser_tgb_up_nd.at[tgb], tgb_up_nd)),
                        dtype=np.int)
            # if cell is a dummy iterate downstream as long as cell is of
            # dummy type and add real cells to first real cell met
            else:
                while ser_tgb_type_dummy.at[tgb_down] \
                    and ser_tgb_down.at[tgb_down] != np.max(ser_tgb_down.index.values):
                    tgb_down = ser_tgb_down.at[tgb_down]
                ser_tgb_up_nd.at[tgb_down] = np.sort(np.array(
                        np.concatenate((ser_tgb_up_nd.at[tgb_down], tgb_up_nd)),
                        dtype=np.int))
    return ser_tgb_up_nd

# %% calculate slope for routing
def calc_ch_sl(ser_ch_zdif, ser_ch_fl, ser_tgb_type_routing, def_sl_excl_quant=None):
    """
    This function calculates the channel elevation differences and corrects them
    applying the LARSIM conventions. Tis means, that (1) a minimum channel slope
    is maintained. The slope value might be very small, but is not allowed to be
    zero. As there are LARSIM-internal rounding mechanisms, slope values smaller
    0.0001 mL/mZ have to be neglected. Additionally, (2) multiple upstream 
    neighbour elements have to be balanced, as only one elevation value can be
    applied to a single element. Potential conservation is achieved moving the
    elevation difference to the upstream element neighbours.

    JM 2021

    Arguments:
    -----------
    ser_ch_zdif: pandas.Series [m]
        Series of model elements' channel elevation difference corresponding to
        the serie's ascending index. 
        (e.g., pd.Series([22.4, 12.0,  2.5, 13.8],
                         index=[1, 2, 3, 4], name='ser_ch_zdif'))
    ser_ch_fl: pandas.Series
        Series of model elements' channel flow length corresponding to the serie's
        ascending index. [m]
        (e.g., pd.Series([308.4, 341.0, 204.5, 133.8],
                         index=[1, 2, 3, 4], name='ser_ch_fl'))
    ser_tgb_type_routing: pandas.Series
        Boolean Series, which identifies the routing cells corresponding to the
        serie's ascending index with True.
        (e.g., pd.Series(data=[0, 0, 1, 1], index=[1, 2, 3, 4],
                         name='routing', dtype='bool'))
    def_sl_excl_quant: float (optional, default: None)
        quantile of slope values to be set constant to quantile value
        (e.g., 0.999 sets the upper 0.1% of the slope values to the 0.1% quantile value)

    Returns:
    -----------
    ser_zlower: pandas.Series [m]
        Series of model elements' channel slope corresponding to the serie's
        ascending index. 
    """    
    # calculate slope
    ser_ch_sl = ser_ch_zdif / ser_ch_fl
    # correct unrealistic high slope values (if defined)
    if def_sl_excl_quant:
        def_sl_upper_thr = ser_ch_sl.quantile(def_sl_excl_quant)
        ser_ch_sl.at[ser_ch_sl > def_sl_upper_thr] = def_sl_upper_thr
    # set all elements except routing elements to nan
    ser_ch_sl.at[~ser_tgb_type_routing] = np.nan
    ser_ch_sl.name = 'gef'
    return ser_ch_sl

# %% export parameters to table
def df_to_table(df_dat, path_gdb_out, name_par_tab):
    """
    This function writes a pandas.DataFrame with all necessary parameters to
    an ArcGIS table. 

    JM 2021

    Arguments:
    -----------
    df_dat: pandas.DataFrame
        DataFrame of all parameters, which are needed in the resulting file.
    path_gdb_out: str
        path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
    name_par_tab: str
        file name, where the table shall be stored (e.g., 'tgb_par_tab'))
    
    Returns:
    -----------
    Saves an ArcGIS table with the provided information
    
    """
    # convert DataFrame to structured numpy array (thanks to USGS:
    # https://my.usgs.gov/confluence/display/cdi/pandas.DataFrame+to+ArcGIS+Table)
    structarr_data_tgbdat = np.array(np.rec.fromrecords(df_dat.values))
    names = df_dat.dtypes.index.tolist()
    structarr_data_tgbdat.dtype.names = tuple(names)
    
    # export table
    path_tgb_par_tab = path_gdb_out + name_par_tab
    # delete table if existing
    if arcpy.Exists(path_tgb_par_tab):
        arcpy.Delete_management(path_tgb_par_tab)
    # create table
    arcpy.da.NumPyArrayToTable(structarr_data_tgbdat, path_tgb_par_tab)

# %% write tgb.dat file
def write_tgbdat(df_data_tgbdat, path_tgbdat, def_tgb_nodata_val=-1, 
                 hcs_epsg=31467, vcs_unit='m ue. NN', src_geodata='',
                 catch_name='', comment='', 
                 print_out=False):
    """
    This function writes a pandas.DataFrame with all necessary parameters to
    the spatial LARSIM input file tgb.dat. 

    JM 2021

    Arguments:
    -----------
    df_data_tgbdat: pandas.DataFrame
        DataFrame of all parameters, which are needed in the resulting file.
        The DataFrame includes the model element ID as index and the following columns:
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
    path_tgbdat: str
        path including file name, where the file tgb.dat shall be stored
        (e.g., 'c:\folder\tgb.dat')
    def_tgb_nodata_val: integer (optional, default: -1)
        three character value representing nodata (or NaN) in the resulting file [-]
    hcs_epsg: integer (optional, default: 31467 = GK4)
        EPSG code representing the horizontal coordinate system of elements'
        x- and y-coordinates
    vcs_unit: integer (optional, default: 'm ue. NN' = meter above sea level)
        string representing the vertical coordinate system of elevation values
    src_geodata: str (optional, default: '')
        string describing the source of the used geodata
    catch_name: str (optional, default: '')
        name of the catchment, that the model is for
    comment: str (optional, default: '')
        additional text, that shall describe something in the model
    print_out: boolean (optional, default: '')
        true if workprogress shall be print to command line
    
    Returns:
    -----------
    Saves the spatial model input file tgb.dat
    
    """
    # check if given coordinate system EPSG code is allowed in LARSIM
    hcs_df = pd.DataFrame([
            [31466, 'DHDN / Gauß-Krüger Zone 2',            'm'],
            [31467, 'DHDN / Gauß-Krüger Zone 3',            'm'],
            [31468, 'DHDN / Gauß-Krüger Zone 4',            'm'],
            [31469, 'DHDN / Gauß-Krüger Zone 5',            'm'],
            [21781, 'CH1903 (Schweizer Koordinatensystem)', 'm'],
            [ 4326, 'WGS-84 / geographisch 2D',             'm'],
            [25832, 'ETRS89 / UTM Zone 32N',                'm'],
            [25833, 'ETRS89 / UTM Zone 33N',                'm'],
            [31254, 'Austria GK West',                      'm'],
            [31257, 'Austria GK M28',                       'm']], columns=['EPSG', 'hcs_name', 'unit'])
    if np.isin(hcs_epsg, hcs_df.EPSG):
        if print_out: print('   coordinate system recognized: {0:s} (EPSG: {1:d})...'.format(
                hcs_df.loc[hcs_df.EPSG==hcs_epsg, 'hcs_name'].iloc[0], hcs_epsg))
    else:
        print('ERROR: unknown corrdinate system EPSG-Code {0:d}! Choose of the following:'.format(hcs_epsg))
        print(hcs_df.to_string(header=['EPSG', 'name', 'unit'], index=False))
        sys.exit([0])
    # check if given height system is allowed in LARSIM
    vcs_df = pd.DataFrame({
            'WKID': [7837],
            'vcs_name': ['DHHN2016_(height)'],
            'unit': ['m ue. NN']})
    if np.isin(vcs_unit, vcs_df.unit):
        if print_out: print('   vertical coordinate system recognized: {0:s}...'.format(vcs_unit))
    else: 
        print('ERROR: unknown vertical coordinate system {0:s}! Choose of the following:'.format(vcs_unit))
        print(vcs_df.to_string(header=['WKID', 'name', 'unit'], index=False))
        sys.exit([0])
        
    # pre-define formats for variables
    fields = ['TGB', 'NRFLV', 'FT', 'TAL', 'HUT', 'HOT', 'X', 'Y', 'KMO', 'KMU',
              'GEF', 'HM', 'BM', 'BL', 'BR', 'BBL', 'BBR', 'BNM', 'BNL', 'BNR',
              'BNVRL', 'BNVRR', 'SKM', 'SKL', 'SKR', 'Kommentar_EZG-A', 'Kommentar_GBA']
    print_fmt = pd.DataFrame(np.array([
            [5,  7, 6, 7, 7, 7, 8, 8, 7, 7, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 16, 14],
            [0, -1, 3, 3, 1, 1, 0, 0, 0, 0, 4, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  2,  3]]).T,
        columns=['digits', 'decimals'],
        index=fields)
    # define heading comment
    if catch_name: tgb_catchm_name = '{0:s}'.format(catch_name)
    else: tgb_catchm_name = ''
    tgb_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    tgb_comment = ('#\r'
                   '# tgb.dat: automated production using TATOO package\r'
                   '# author: J. Mitterer, Chair of Hydrology and RBM, Technical University Munich\r'
                   '# geodata: {0:s}\r'
                   '# {1:s}\r'
                   '# time: {2:s}\r'
                   '# comment: {3:s}\r'
                   '#\r').format(src_geodata, tgb_catchm_name, tgb_timestamp, comment)
    # define LARSIM LILA key words
    tgb_keywords = (
            'Koordinatensystem: {0:d}\r'
            'Hoehensystem: {1:s}\r'.format(hcs_epsg, vcs_unit))
    # define variable titles
    tgb_var_titles = ''
    dig_fmt = '{0:s}'
    for var, dig in print_fmt.digits.iteritems():
        tgb_var_titles = tgb_var_titles + dig_fmt.format(var.rjust(dig) + ';')
    # summarize header lines
    tgb_header = tgb_comment + tgb_keywords + tgb_var_titles[:-1]
    # calculate data formats
    fmt_str = []
    for var, (dig, dec) in print_fmt.iterrows():
        if dec == 0: fmt_str.append('%{0:d}d'.format(dig))
        elif dec < 0: fmt_str.append('%{0:d}s'.format(dig))
        else: fmt_str.append('%{0:d}.{1:d}f'.format(dig, dec))
    # write data to file
    np.savetxt(path_tgbdat, df_data_tgbdat.loc[:,fields],
               delimiter=';', fmt=fmt_str, newline=';\r',
               header=tgb_header, footer='', comments='')
    # import written text file and replace NaN with -1 
    fid = open(path_tgbdat, 'r')
    tgb_str = fid.read()
    tgb_str = tgb_str.replace('nan', str(def_tgb_nodata_val).rjust(3))
    fid.close()
    # write corrected file string to same file
    fid = open(path_tgbdat, 'w')
    fid.write(tgb_str)
    fid.close()
    
# %% write utgb.dat file
def write_utgbdat(df_data_utgbdat, path_utgbdat, 
                  ctrl_opt_infdyn, ctrl_opt_impperc, ctrl_opt_capr, ctrl_opt_siltup,
                  udef_tgb_nodata_val=-1, src_geodata='', catch_name='', comment='', 
                  print_out=False):
    r"""
    This function writes a pandas.DataFrame with all necessary parameters to
    the spatial LARSIM input file tgb.dat. 

    JM 2021

    Arguments:
    -----------
    df_data_utgbdat: pandas.DataFrame
        DataFrame of all parameters, which are needed in the resulting file.
        The DataFrame has to include the following columns:
        - 'TGB': model element ID number (int)
        - 'UTGB': hru ID number within model element (int)
        - 'LN': land use ID number (int)
        - 'Flaeche': hru area [km²] (float)
        - 'nFK': usable field capacity [mm] (int)
        - 'LK': air capacity [mm] (int)
        The DataFrame may include the following columns:
        using defined impervios shares:
        - 'Vgrad': share of impervious area [%] (int)
        using dynamic infiltration (INFILTRATION DYNAMISCH):
        - 'nFKVol': usable field capacity within upper soil layer [Vol-%] (float)
        - 'LKVol': air capacity within upper soil layer [Vol-%] (float)
        - 'ks': saturated conductivity [mm/h] (float)
        - 'wsf': suction at the wetting front [mm] (float)
        - 'MPdi': density of macro pores [#/m²] (float)
        - 'MPla': depth of macro pores [mm] (float)
        - 'TRti': depth of drying cracks [mm] (float)
        - 'AuGr': water content at lower plasticity threshold [% of nFK] (float)
        - 'SchrGr': water content at lower shrinkage threshold [% of nFK] (float)
        using dynamic infiltration (INFILTRATION DYNAMISCH) with silting up:
        - 'VF': infiltration reduction factor for silting up [-] (float)
        using capillary rise (KAPILLARER AUFSTIEG or KOPPLUNG BODEN/GRUNDWASSER):
        - 'KapA': capacity of capillary rise [mm/h] (float)
    path_utgbdat: str
        path including file name, where the file tgb.dat shall be stored
        (e.g., 'c:\folder\utgb.dat')
    ctrl_opt_infdyn: boolean
        control operator to activate dynamic infiltration parametrization
    ctrl_opt_impperc: boolean
        control operator to activate sealing parametrization
    ctrl_opt_capr: boolean
        control operator to activate capillary rise parametrization
    ctrl_opt_siltup: boolean
        control operator to activate silting-up parametrization
    udef_tgb_nodata_val: integer (optional, default: -1)
        three character value representing nodata (or NaN) in the resulting file [-]
    src_geodata: str (optional, default: '')
        string describing the source of the used geodata
    catch_name: str (optional, default: '')
        name of the catchment, that the model is for
    comment: str (optional, default: '')
        additional text, that shall describe something in the model
    print_out: boolean (optional, default: '')
        true if workprogress shall be print to command line
    
    Returns:
    -----------
    Saves the spatial model input file tgb.dat
    """
    if print_out: print(r'...write utgb.dat file...')
    # reorder columns for output
    field_list = ['TGB', 'UTGB', 'LN', 'Flaeche', 'nFK', 'LK']
    digits   = [5, 5, 5, 11, 5, 5]
    decimals = [0, 0, 0,  8, 0, 0]
    if ctrl_opt_infdyn:  
        field_list += ['nFKVol', 'LKVol', 'ks'  , 'wsf'   , 'MPdi',
                       'MPla'  , 'TRti' , 'AuGr', 'SchrGr']
        digits     += [7, 7, 7, 7, 7, 7, 7, 7, 7]
        decimals   += [1, 1, 1, 1, 1, 1, 1, 1, 1]
    if ctrl_opt_impperc:
        field_list += ['Vgrad']
        digits     += [6]
        decimals   += [0]
    if ctrl_opt_capr:
        field_list += ['KapA']
        digits     += [5]
        decimals   += [1]
    if ctrl_opt_siltup:
        field_list += ['VF']
        digits     += [5]
        decimals   += [1]
    df_hru = df_data_utgbdat[field_list].astype(np.float)
    df_hru = df_hru.astype({'TGB': np.int, 'UTGB': np.int, 'LN': np.int,
                            'nFK': np.int, 'LK'  : np.int})    
    print_fmt = pd.DataFrame(np.array([digits, decimals]).T,
        columns=['digits', 'decimals'],
        index=field_list)
    # define heading comment
    if catch_name: catchm_name_str = '{0:s}'.format(catch_name)
    else: catchm_name_str = ''
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    comment_str = ('#\r'
                   '# utgb.dat: automated production using TATOO package\r'
                   '# author: J. Mitterer, Chair of Hydrology and RBM, Technical University Munich\r'
                   '# geodata: {0:s}\r'
                   '# {1:s}\r'
                   '# time: {2:s}\r'
                   '# comment: {3:s}\r'
                   '#\r').format(src_geodata, catchm_name_str, timestamp_str, comment)
    # define LARSIM LILA key words
    max_hrus_per_el = np.int32(np.max(df_data_utgbdat.loc[:, 'UTGB']))
    utgb_keywords = (
            'Maximale Anzahl an Unterteilgebieten: {0:d}\r'.format(max_hrus_per_el))
    # define variable titles
    var_titles_str = ''
    dig_fmt = '{0:s}'
    for var, dig in print_fmt.digits.iteritems():
        var_titles_str = var_titles_str + dig_fmt.format(var.rjust(dig) + ';')
    # summarize header lines
    header_str = comment_str + utgb_keywords + var_titles_str[:-1]
    # calculate data formats
    fmt_str = []
    for var, (dig, dec) in print_fmt.iterrows():
        if dec == 0: fmt_str.append('%{0:d}d'.format(dig))
        elif dec < 0: fmt_str.append('%{0:d}s'.format(dig))
        else: fmt_str.append('%{0:d}.{1:d}f'.format(dig, dec))
    # write data to file
    np.savetxt(path_utgbdat, df_data_utgbdat.loc[:,field_list],
               delimiter=';', fmt=fmt_str, newline=';\r',
               header=header_str, footer='', comments='')
    # import written text file and replace NaN with -1 
    fid = open(path_utgbdat, 'r')
    utgb_str = fid.read()
    utgb_str = utgb_str.replace('nan', str(udef_tgb_nodata_val).rjust(3))
    fid.close()
    # write corrected file string to same file
    fid = open(path_utgbdat, 'w')
    fid.write(utgb_str)
    fid.close()
    
# %% calculate HRUs' parameters based on selected GIS data and methods
def calc_hrus(path_tgb_sj, path_soil, path_lu, f_tgb, f_lu_id,
              lu_id_imp, lu_id_water, def_amin_utgb_del,
              path_gdb_out, name_hru_c='hru_c', 
              ctrl_opt_impperc=False, f_impperc='', 
              ctrl_opt_infdyn=False, df_lu_mp=None,
              ctrl_opt_siltup=False,
              ctrl_opt_capr=False, 
              ctrl_hru_aggr=False,
              print_out=False):
    """
    This function calculate and imports HRUs from GIS data and converts it to a
    pandas DataFrame. It processes the following steps:
        - check the defined GIS input data for necessary fields dependent on the
          activated methods (ctrl_opt_impperc, ctrl_opt_infdyn, ctrl_opt_siltup,
          ctrl_opt_capr)
        - intersect model element, soil and land use GIS data to a feature class
        - if activated (ctrl_hru_aggr), aggregate HRUs per model element summing
          up the area (and averaging the impervious share if ctrl_opt_impperc
          is activated).
        - if activated (ctrl_opt_infdyn), calculate the macropore parameters
          macropore density and length
    
    JM 2021
    
    Arguments:
    -----------
    General arguments:
        path_tgb_sj, path_soil, path_lu: str
            input paths:
                - tgb_sj: polygon feature class of model elements
                - soil: polygon feature class of land use data and sealing percentage
                - lu: polygon feature class of soil data
        f_tgb, f_lu_id: str (e.g., 'tgb' and 'landuse_id')
            string representing the GIS fields for the model element ID (f_tgb)
            and the land use ID (f_lu_id).
        lu_id_imp, lu_id_water: int
            land use ID numbers for 'impervious' and 'water' land use classes
        def_amin_utgb_del: float
            area threshold below which HRUs are deleted
        path_gdb_out: str
            path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
        name_hru_c: str (optional, default: 'hru_c')
            output path of polygon feature class of HRUs
        f_a: str (optional, default: 'area')
            string representing the HRU field for the area (f_a)
    Arguments for specific methods:
        ctrl_opt_impperc: boolean (optional, default: False)
            (de-)activate the import of user-defined, spatial, impervious share
        f_impperc: str (optional, default: '')
            string representing the GIS field for the impervious share.
        ctrl_opt_infdyn: boolean (optional, default: False)
            (de-)activate the import of soil information needed for dynamic
            infiltration (metod: INFILTRATION DYNAMISCH)
        df_lu_mp: pandas.DataFrame (optional, default: None)
            dataframe containing landuse-dependent (df.index) parameters for
            macropore density (MPla) and length (MPla)
        ctrl_opt_siltup: boolean (optional, default: False)
            (de-)activate the import of silting-up (metod: INFILTRATION DYNAMISCH)
        ctrl_opt_capr: boolean (optional, default: False)
            (de-)activate the import of soil information needed for capillary
            rise (metods: KAPILLARER AUFSTIEG or KOPPLUNG BODEN/GRUNDWASSER)
        ctrl_hru_aggr: boolean (optional, default: False)
            (de-)activate the aggrigation of HRUs within each model element. 
            Dependent on the number of HRUs, the aggregation might significally
            decrease the simulation time of LARSIM.
    Print arguments:
        print_out: boolean (optional, default: '')
            true if workprogress shall be print to command line
    
    Returns:
    -----------
    The function saves an output polygon feature class (name_hru_c)
    df_hru: pandas.DataFrame
        HRU DataFrame, which includes all fields defined with booleans. 
    """
    
    # %% calculate and impoert HRUs from model element, soil and land use GIS data
    def calc_hrus_from_gis(path_tgb_sj, path_soil, path_lu, f_tgb, f_lu_id,
                           path_gdb_out, name_hru_c='hru_c', f_a='area', 
                           ctrl_opt_impperc=False, f_impperc='', f_vgrad='Vgrad',
                           ctrl_opt_infdyn=False, df_lu_mp=None,
                           ctrl_opt_siltup=False,
                           ctrl_opt_capr=False, 
                           ctrl_hru_aggr=False,
                           print_out=False):
        """
        This function calculate and imports HRUs from GIS data and converts it to a
        pandas DataFrame. It processes the following steps:
            - check the defined GIS input data for necessary fields dependent on the
              activated methods (ctrl_opt_impperc, ctrl_opt_infdyn, ctrl_opt_siltup,
              ctrl_opt_capr)
            - intersect model element, soil and land use GIS data to a feature class
            - if activated (ctrl_hru_aggr), aggregate HRUs per model element summing
              up the area (and averaging the impervious share if ctrl_opt_impperc
              is activated).
            - if activated (ctrl_opt_infdyn), calculate the macropore parameters
              macropore density and length
        
        JM 2021
        
        Arguments:
        -----------
        General arguments:
            path_tgb_sj, path_soil, path_lu: str
                input paths:
                    - tgb_sj: polygon feature class of model elements
                    - soil: polygon feature class of land use data and sealing percentage
                    - lu: polygon feature class of soil data
            f_tgb, f_lu_id: str (e.g., 'tgb' and 'landuse_id')
                string representing the GIS fields for the model element ID (f_tgb)
                and the land use ID (f_lu_id).
            path_gdb_out: str
                path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
            name_hru_c: str (optional, default: 'hru_c')
                output path of polygon feature class of HRUs
            f_a: str (optional, default: 'area')
                string representing the HRU field for the area (f_a)
        Arguments for specific methods:
            ctrl_opt_impperc: boolean (optional, default: False)
                (de-)activate the import of user-defined, spatial, impervious share
            f_impperc: str (optional, default: '')
                string representing the GIS field for the impervious share.
            f_vgrad: str (optional, default: 'Vgrad')
                string representing the HRU table field for the impervious share.
                This is a parameter whose name is recognized by LARSIM. 
                --> do not change!
            ctrl_opt_infdyn: boolean (optional, default: False)
                (de-)activate the import of soil information needed for dynamic
                infiltration (metod: INFILTRATION DYNAMISCH)
            df_lu_mp: pandas.DataFrame (optional, default: None)
                dataframe containing landuse-dependent (df.index) parameters for
                macropore density (MPla) and length (MPla)
            ctrl_opt_siltup: boolean (optional, default: False)
                (de-)activate the import of silting-up (metod: INFILTRATION DYNAMISCH)
            ctrl_opt_capr: boolean (optional, default: False)
                (de-)activate the import of soil information needed for capillary
                rise (metods: KAPILLARER AUFSTIEG or KOPPLUNG BODEN/GRUNDWASSER)
            ctrl_hru_aggr: boolean (optional, default: False)
                (de-)activate the aggrigation of HRUs within each model element. 
                Dependent on the number of HRUs, the aggregation might significally
                decrease the simulation time of LARSIM.
        Print arguments:
            print_out: boolean (optional, default: '')
                true if workprogress shall be print to command line
        
        Returns:
        -----------
        The function saves an output polygon feature class (name_hru_c)
        df_hru: pandas.DataFrame
            HRU DataFrame, which includes all fields defined with booleans. 
        """
        # internal functions
        def del_all_fields_except(path_table, exclude_field_list):
            """
            function to delete all fields in an ArcGIS table except user-defined list
            """
            delete_field_list = []
            for field in arcpy.ListFields(path_table):
                if not field.required and not field.name in exclude_field_list:
                    delete_field_list.append(field.name)
            if len(delete_field_list) > 0:
                arcpy.DeleteField_management(path_table, delete_field_list)
                
        def check_fields(path_table, field_list):
            """
            function to check existance of fields in an ArcGIS table
            """
            gis_field_list = [field.name for field in arcpy.ListFields(path_table)]
            fields_notincl = [x for x in field_list if x not in gis_field_list]
            if len(fields_notincl) > 0:
                sys.exit(('The feature class does not contain all fields needed: '
                          '{0}').format(fields_notincl))
        
        # definitions
        arcpy.env.workspace = path_gdb_out
        # define intermediate ArcGIS feature class names
        name_soil_c   = 'soil_c'
        name_lu_c     = 'landuse_c'
        name_tgb_copy = 'tgb_copy'
        name_hru_tab  = 'hru_aggr_tab'
        # ArcGIS field definitions
        f_oid     = 'OBJECTID'
        f_shp_a   = 'Shape_Area'
        f_sum     = 'SUM'
        f_mean    = 'MEAN'
        f_freq    = 'FREQUENCY'
        # field lists
        field_list_tgb_standard  = [f_tgb]
        field_list_soil_standard = ['nFK', 'LK']
        field_list_soil_infdyn   = ['nFKVol', 'LKVol', 'ks', 'wsf',
                                    'TRti'  , 'AuGr' , 'SchrGr']
        field_list_soil_capr     = ['KapA']
        field_list_soil_siltup   = ['VF']
        field_list_lu_standard   = [f_lu_id]
        field_list_lu_imp        = [f_impperc]
        
        # caclulations
        # To allow overwriting the outputs change the overwrite option to true.
        arcpy.env.overwriteOutput = True
        # copy and prepare TGB shape layer
        if print_out: print('...copy model structure layer...')
        # check necessary fields
        imp_field_list = copy.deepcopy(field_list_tgb_standard)
        check_fields(path_tgb_sj, field_list_tgb_standard)
        # copy feature class
        path_tgb_copy = path_gdb_out + name_tgb_copy
        arcpy.management.CopyFeatures(path_tgb_sj, path_tgb_copy, '', None, None, None)
        # delete fields, which are not necessary
        del_all_fields_except(path_tgb_copy, field_list_tgb_standard)
        # clip and prepare soil layer
        if print_out: print('...clip and prepare soil layer...')
        # define necessary fields
        exclude_field_list = copy.deepcopy(field_list_soil_standard)
        if ctrl_opt_infdyn: exclude_field_list += field_list_soil_infdyn
        if ctrl_opt_capr:   exclude_field_list += field_list_soil_capr
        if ctrl_opt_siltup: exclude_field_list += field_list_soil_siltup
        imp_field_list += exclude_field_list
        # check necessary fields
        check_fields(path_soil, exclude_field_list)
        # clip feature class
        path_soil_c = path_gdb_out + name_soil_c
        arcpy.analysis.Clip(path_soil, path_tgb_copy, path_soil_c, None)
        # remove not necessary fields according to the option specifications
        del_all_fields_except(path_soil_c, exclude_field_list)
        # clip and prepare soil layer
        if print_out: print('...clip and prepare land use layer...')
        # define necessary fields
        exclude_field_list = copy.deepcopy(field_list_lu_standard)
        if ctrl_opt_impperc: exclude_field_list += field_list_lu_imp
        imp_field_list += exclude_field_list
        # check necessary fields
        check_fields(path_lu, exclude_field_list)
        # clip feature class
        path_lu_c = path_gdb_out + name_lu_c
        arcpy.analysis.Clip(path_lu, path_tgb_copy, path_lu_c, None)
        # remove not necessary fields according to the option specifications
        del_all_fields_except(path_lu_c, exclude_field_list)
        # Intersect soil, land use and model raster layers
        if print_out: print('...intersect soil, land use and model raster layers...')
        path_hru_c = path_gdb_out + name_hru_c
        arcpy.analysis.Intersect(path_tgb_copy + ' #;' + path_soil_c + ' #;' + path_lu_c + ' #',
                                 path_hru_c, 'NO_FID', None, 'INPUT')
        # aggregate HRUs if necessary
        if ctrl_hru_aggr:
            if print_out: print('...aggregate HRUs...')
            # get list of indipendent fields
            aggr_field_list = [x for x in imp_field_list if x not in [f_shp_a, f_impperc]]
            # aggregate HRUs
            # define output path of HRU table
            path_hru_tab = path_gdb_out + name_hru_tab
            # define expression for field aggregation
            aggr_expr = '{0} {1}'.format(f_shp_a, f_sum)
            if ctrl_opt_impperc:
                aggr_expr += ';{0} {1}'.format(f_impperc, f_mean)
            # calculate statistics
            arcpy.analysis.Statistics(path_hru_c, path_hru_tab, aggr_expr,
                                      ';'.join(aggr_field_list))
            # delete not needed fields
            arcpy.management.DeleteField(path_hru_tab, f_freq)
            # alter statistical result fields
            arcpy.management.AlterField(
                            path_hru_tab, '{0}_{1}'.format(f_sum, f_shp_a),
                            f_a, '', 'DOUBLE', 8, 'NULLABLE', 'CLEAR_ALIAS')
            if ctrl_opt_impperc:
                arcpy.management.AlterField(
                                path_hru_tab, '{0}_{1}'.format(f_mean, f_impperc),
                                f_vgrad, '', 'DOUBLE', 8, 'NULLABLE', 'CLEAR_ALIAS')
            # import structured numpy array
            if print_out: print('...import HRUs...')
            arr_hru = arcpy.da.FeatureClassToNumPyArray(path_hru_tab, '*')
            # convert numpy array to pandas DataFrame
            if ctrl_opt_impperc:
                imp_field_list.remove(f_impperc)
                imp_field_list.append(f_vgrad)
            imp_field_list.append(f_a)
            df_hru = pd.DataFrame(arr_hru[imp_field_list], index=arr_hru[f_oid])
        else:
            # import structured numpy array and convert to DataFrame
            if print_out: print('...import HRUs...')
            arr_hru = arcpy.da.FeatureClassToNumPyArray(path_hru_tab, '*')
            # convert numpy array to pandas DataFrame
            imp_field_list += [f_shp_a]
            df_hru = pd.DataFrame(arr_hru[imp_field_list], index=arr_hru[f_oid])
            # rename variables
            df_hru = df_hru.rename(columns={f_shp_a:   f_a,
                                            f_impperc: f_vgrad})
        
        # calculate macro pore parameters
        if ctrl_opt_infdyn:
            print('...calculate macro pore parameters...')
            # calculate macropore density and length
            df_hru.at[:, 'MPdi'] = df_lu_mp.loc[df_hru.loc[:, f_lu_id], 'MPdi'].values
            df_hru.at[:, 'MPla'] = df_lu_mp.loc[df_hru.loc[:, f_lu_id], 'MPla'].values    
        
        return df_hru
    
    # %% data corrections with LARSIM conventions
    def corr_f_vals(df_hru, lu_id_imp=3, lu_id_water=16, def_amin_utgb_del=10**-8,
                    f_a='area', f_lu_id='landuse_id', 
                    ctrl_opt_impperc=False, f_vgrad='Vgrad'):
        """
        LARSIM conventions set some strict rules, HRU parameters have to follow.
        Thi function allows for automatic control and, if necessary, corrections.
        The following issues are generally handled:
            - if any element's soil storages (usable field or air capacity) are
              zero, but they are neither of land use class 'water' nor 'impervious'
              --> break with ERROR
            - if there are HRUs with an area less than LARSIM internal rounding
              value --> delete HRUs
            - if there are HRUs with land uses 'water' and 'impervious' combined
              with soil property values not-equal to zero --> set soil property 
              fields to zero
        If user-defined impervious share shall be given, the following issues are
        additionally handled:
            - if impervious percentage is 100 % but land use class is not
              'impervious' --> set impervious percentage to 99 %
            - if impervious percentage is less than 100 % but land use class is
              'impervious' --> set impervious percentage to 100 %
            - if impervious percentage is not 0 % but land use class is
              'water' --> set impervious percentage to 0 %
        
        JM 2021
        
        Arguments:
        -----------
        df_hru: pandas.DataFrame
            input HRU DataFrame. It has to include the fields specified in the 
            function input (f_a, f_lu_id, and optional f_vgrad). All other
            field values are set to zero. 
        lu_id_imp, lu_id_water: int (optional, default: 3, 16)
            land use classes 'impervious' and 'water'
        def_amin_utgb_del: float (optional, default: 10**-8)
            area threshold below which HRUs are deleted
        f_a, f_lu_id, f_vgrad: str (optional, default: 'area', 'landuse_id', 'Vgrad')
            string representing the HRU field for the area (f_a), the land use
            class ID (f_lu_id), and user-defined impervious share (f_vgrad)
        ctrl_opt_impperc: boolean (optional, default: False)
            switch to (de-)activate the option for user-defined impervious share
        
        Returns:
        -----------
        df_hru: pandas.DataFrame
            updated HRU DataFrame. It includes all fields defined for the input HRU
            DataFrame. 
        """
        # break, if any element's soil storages (usable field or air capacity) are
        # zero, but they are neither of land use class 'water' nor 'impervious'
        hrus_soil_mism_bool = np.logical_and(
                np.logical_or(df_hru.loc[:, 'nFK'] == 0,
                              df_hru.loc[:, 'LK' ] == 0),
                ~ np.isin(df_hru.loc[:, f_lu_id], [lu_id_water, lu_id_imp]))
        if np.any(hrus_soil_mism_bool):
            print('ERROR: There are HRUs, whose soil storages (usable field or air'
                  'capacity) are zero, but they are neither of land use class'
                  '"water" nor "impervious":')
            print(df_hru.loc[hrus_soil_mism_bool, :])
            sys.exit()
    
        # remove hrus with an area less than LARSIM internal rounding
        # recalculate area from [m²] to [km²]
        df_hru.loc[:, f_a] = df_hru.loc[:, f_a] / 10**6
        # remove UTGBs with area < def_amin_utgb_del [km²]
        if np.any(df_hru.loc[:, f_a] < def_amin_utgb_del):
            print('...remove extremely small HRUs...')
            df_hru = df_hru.drop(df_hru[df_hru.loc[:, f_a] >= def_amin_utgb_del].index)
        
        # set soil property fields to zero for land uses 'water' and 'impervious'
        hrus_water_ii = np.isin(df_hru.loc[:, f_lu_id], [lu_id_water, lu_id_imp])
        list_f_pot = ['nFK', 'LK', 'wsf', 'SchrGr', 'AuGr', 'nFKVol', 'LKVol',
                      'TRti', 'ks', 'VF', 'MPla', 'MPdi', 'KapA', 'VF', f_vgrad]
        list_f_setnull = [x for x in list_f_pot if x in df_hru.columns]
        df_hru.loc[hrus_water_ii, list_f_setnull] = 0
    
        # correct impervious percentage according to LARSIM conventions
        if ctrl_opt_impperc:
            # if impervious percentage is 100 % but land use class is not
            # 'impervious', set impervious percentage to 99 %
            hrus_i100_luni_b = np.logical_and(df_hru[f_vgrad] == 100,
                                              df_hru[f_lu_id]   != lu_id_imp)
            if np.any(hrus_i100_luni_b):
                print('WARNING: There are HRUs, whose impervious share is 100 %, but '
                      'their land use class is not "impervious":')
                print(df_hru.loc[hrus_i100_luni_b, :])
                print('Their impervious share is set to 99 %.\n')
                df_hru.loc[hrus_i100_luni_b, f_vgrad] = 99
            # if impervious percentage is less than 100 % but land use class is
            # 'impervious', set impervious percentage to 100 %
            hrus_le100_luei_b = np.logical_and(df_hru[f_vgrad] < 100,
                                              df_hru[f_lu_id]   == lu_id_imp)
            if np.any(hrus_le100_luei_b):
                print('WARNING: There are HRUs, whose impervious share is < 100 %, but '
                      'their land use class is "impervious":')
                print(df_hru.loc[hrus_le100_luei_b, :])
                print('Their impervious share is set to 100 %.\n')
            df_hru.loc[hrus_le100_luei_b, f_vgrad] = 100
            # if impervious percentage is not 0 % but land use class is
            # 'water', set impervious percentage to 0 %
            hrus_ne0_lueqw_b = np.logical_and(df_hru[f_vgrad] != 0,
                                              df_hru[f_lu_id] == lu_id_water)
            if np.any(hrus_ne0_lueqw_b):
                print('WARNING: There are HRUs, whose impervious share is not 0 %, but '
                      'their land use class is "water":')
                print(df_hru.loc[hrus_ne0_lueqw_b, :])
                print('Their impervious share is set to 0 %.\n')
                df_hru.loc[hrus_ne0_lueqw_b, f_vgrad] = 0
            
        return df_hru
    
    # %% aggregate HRUs dependent on land use class
    def aggr_lu_hrus(df_hru, lu_id,
                     f_a='area', f_tgb='tgb', f_lu_id='landuse_id',
                     ctrl_opt_impperc=False, f_vgrad='Vgrad', lu_impperc=0):
        """
        LARSIM does not allow more than one HRU of land uses 'impervious' and 'water'
        in a single model element. Therefore, HRUs from GIS import have to be 
        aggregated if there are more than one. This function aggregates HRUs of a 
        defined land use to a single HRU. If a user-defined impervious share shall
        be defined in the HRUs, the user may set the value accordingly.
        
        JM 2021
        
        Arguments:
        -----------
        df_hru: pandas.DataFrame
            input HRU DataFrame. It has to include the fields specified in the 
            function input (f_a, f_tgb, f_lu_id, and optional f_vgrad). All other
            field values are set to zero. 
        lu_id: int
            land use class for which HRUs shall be aggregated
        f_a, f_tgb, f_lu_id: str (optional, default: 'area', 'tgb', 'landuse_id')
            string representing the HRU field for the area (f_a), the model element
            ID (f_tgb), and the land use class ID (f_lu_id).
        ctrl_opt_impperc: boolean (optional, default: False)
            switch to (de-)activate the option for user-defined impervious share
        f_vgrad: str (optional, default: 'Vgrad')
            string representing the HRU field for the user-defined impervious share.
        lu_impperc: float (optional, default: 0)
            user-defined spatial impervious share of the defined land use class [%]
        
        Returns:
        -----------
        df_hru_update: pandas.DataFrame
            updated HRU DataFrame. It includes all fields defined for the input HRU
            DataFrame. 
        """
        df_hru_update = copy.deepcopy(df_hru)
        # iterate model elements
        for tgb in np.unique(df_hru_update.loc[:, f_tgb]):
            
            # correct hydrological response units with defined land use class
            # get HRUs with defined land use of the recent model element ID
            hrus_imp = df_hru_update[np.logical_and(df_hru_update.loc[:, f_tgb]   == tgb,
                                             df_hru_update.loc[:, f_lu_id] == lu_id)]
            # if there is more than one HRU of defined class, aggregate to one
            if hrus_imp.shape[0] > 1:
                # create new HRU ID number (max(IDs) + 1)
                hru_id = [np.max(df_hru_update.index) + 1]
                # pre-allocate aggregated HRU element
                hrus_imp_aggr = pd.DataFrame(np.zeros((1, hrus_imp.shape[1])),
                                             index=hru_id,
                                             columns=df_hru_update.columns).astype(
                                                     df_hru_update.dtypes)
                # fill aggregated HRU element
                hrus_imp_aggr.at[hru_id, f_tgb  ] = tgb
                hrus_imp_aggr.at[hru_id, f_lu_id] = lu_id
                hrus_imp_aggr.at[hru_id, f_a    ] = np.sum(hrus_imp.loc[:, f_a])
                # if impervious share is user-defined, set it to 100 %
                if ctrl_opt_impperc: hrus_imp_aggr.at[hru_id, f_vgrad] = lu_impperc
                # merge aggregated HRU element with remaining HRUs
                df_hru_update = pd.concat((
                        df_hru_update.loc[np.logical_or(
                                df_hru_update.loc[:, f_tgb] != tgb,
                                np.logical_and(df_hru_update.loc[:, f_tgb]   == tgb,
                                               df_hru_update.loc[:, f_lu_id] != lu_id)), :],
                        hrus_imp_aggr))
        # sort array according to TGB and LU
        df_hru_update = df_hru_update.sort_values([f_tgb, f_lu_id], axis=0)
        
        return df_hru_update
    
    # %% calculate HRUs' identification numbers (utgb)
    def calc_hru_ids(df_hru, f_tgb='tgb', f_utgb='utgb'):
        """
        LARSIM needs an ascending ID number for all HRUs inside a model element. 
        This function calculates this ID number based on the given HRU data.
        
        JM 2021
        
        Arguments:
        -----------
        df_hru: pandas.DataFrame
            input HRU DataFrame. It has to include the fields specified in the 
            function input (f_tgb, f_utgb).
        f_tgb, f_utgb: str (optional, default: 'tgb', 'utgb')
            string representing the HRU field for the model element ID (f_tgb) and 
            the HRU ID (f_utgb).
        
        Returns:
        -----------
        df_hru: pandas.DataFrame
            updated HRU DataFrame, which includes the HRU ID. 
        """    
        # pre-set hrus' identification numbers with zeros
        df_hru.at[:, f_utgb] = np.zeros((df_hru.shape[0], ), dtype=np.int)
        # iterate model element numbers
        for tgb in np.unique(df_hru.loc[:, f_tgb]):
            # get logical array of hrus with model element number
            hrus_tgb_ii = df_hru.loc[:, f_tgb] == tgb
            # set hrus' identification numbers according to ascending convention
            df_hru.loc[hrus_tgb_ii, f_utgb] = np.arange(1, np.sum(hrus_tgb_ii) + 1)
        return df_hru
    
    # calculations
    # define parameter names
    f_a = 'area'
    f_vgrad = 'Vgrad'
    # calculate and impoert HRUs
    df_hru = calc_hrus_from_gis(
            path_tgb_sj, path_soil, path_lu, f_tgb, f_lu_id,
            path_gdb_out, name_hru_c=name_hru_c, f_a=f_a, 
            ctrl_opt_impperc=ctrl_opt_impperc, f_impperc=f_impperc, f_vgrad=f_vgrad,
            ctrl_opt_infdyn=ctrl_opt_infdyn, df_lu_mp=df_lu_mp,
            ctrl_opt_siltup=ctrl_opt_siltup,
            ctrl_opt_capr=ctrl_opt_capr, 
            ctrl_hru_aggr=ctrl_hru_aggr,
            print_out=print_out)
    # data corrections with LARSIM conventions
    df_hru = corr_f_vals(df_hru, lu_id_imp=lu_id_imp, lu_id_water=lu_id_water,
                         def_amin_utgb_del=def_amin_utgb_del,
                         f_a=f_a, f_lu_id=f_lu_id, 
                         ctrl_opt_impperc=ctrl_opt_impperc, f_vgrad=f_vgrad)
    # aggregate HRUs dependent on land use class
    df_hru = aggr_lu_hrus(df_hru,  3,
                          f_a=f_a, f_tgb=f_tgb, f_lu_id=f_lu_id,
                          ctrl_opt_impperc=ctrl_opt_impperc, f_vgrad=f_vgrad, 
                          lu_impperc=100)
    df_hru = aggr_lu_hrus(df_hru, 16,
                          f_a=f_a, f_tgb=f_tgb, f_lu_id=f_lu_id,
                          ctrl_opt_impperc=ctrl_opt_impperc, f_vgrad=f_vgrad, 
                          lu_impperc=  0)
    # calculate HRUs' identification numbers (utgb)
    f_utgb = 'utgb'
    df_hru = calc_hru_ids(df_hru, f_tgb=f_tgb, f_utgb=f_utgb)
    # create GIS table
    if print_out: print('...create GIS table...')
    # rename fields for output
    df_data_utgbdat = df_hru.rename(columns={f_tgb:  'TGB', f_utgb: 'UTGB',
                                             f_lu_id: 'LN', f_a: 'Flaeche'})
    # reorder columns for output
    field_list = ['TGB', 'UTGB', 'LN', 'Flaeche', 'nFK', 'LK']
    if ctrl_opt_infdyn:  field_list += ['nFKVol', 'LKVol', 'ks'  , 'wsf'   , 'MPdi',
                                        'MPla'  , 'TRti' , 'AuGr', 'SchrGr']
    if ctrl_opt_impperc: field_list += [f_vgrad]
    if ctrl_opt_capr:    field_list += ['KapA']
    if ctrl_opt_siltup:  field_list += ['VF']
    df_data_utgbdat = df_data_utgbdat[field_list].astype(np.float)
    df_data_utgbdat = df_data_utgbdat.astype({'TGB': np.int, 'UTGB': np.int,
                                              'LN': np.int, 
                                              'nFK': np.int, 'LK'  : np.int})
    return df_data_utgbdat

# %% export parameters to point feature class
def tgb_to_points(df_data, sr_obj, path_gdb_out, name_fc,
                  geometry_fields=('x', 'y')):
    """
    This function writes a pandas.DataFrame with all necessary parameters to
    an ArcGIS table. 

    JM 2021

    Arguments:
    -----------
    df_data: pandas.DataFrame
        DataFrame of all parameters, which are needed in the resulting file.
    sr_obj: arcpy.SpatialReferenceObject
        arcpy.Object containing the spatial reference of the final feature class
    path_gdb_out: str
        path of the output file geodatabase (e.g., 'c:\model_creation.gdb')
    name_fc: str (e.g., 'tgb_p')
        file name, where the feature class shall be stored
    geometry_fields: tuple (optional, default: ('x', 'y')
        field name of table and polygon feature class to join attributes
    
    Returns:
    -----------
    Saves an ArcGIS table with the provided information
    """
    # convert DataFrame to structured numpy array (thanks to USGS:
    # https://my.usgs.gov/confluence/display/cdi/pandas.DataFrame+to+ArcGIS+Table)
    structarr_data = np.array(np.rec.fromrecords(df_data.values))
    names = df_data.dtypes.index.tolist()
    structarr_data.dtype.names = tuple(names)
    
    # export feature class
    path_fc = path_gdb_out + name_fc
    # delete feature class if existing
    if arcpy.Exists(path_fc):
        arcpy.Delete_management(path_fc)
    # create point feature class
    arcpy.da.NumPyArrayToFeatureClass(
            structarr_data, path_fc, geometry_fields, sr_obj)

# %% calculate characteristic channel-forming discharge
def calc_ch_form_q(ser_area_infl, ser_tgb_down, q_spec, ser_q_in_corr=None):
    """
    This function calculates the channel forming discharge. In Allen et al. (1994)
    it is reffered to as a discharge occuring once in two years (HQ2), in other
    sources it is estimated as annual high discharge or a discharge occurring
    once in 2.33 years. 
    Using a specific discharge (may be HQ1, HQ2 or HQ2.33), the algorithm defines
    a channel forming discharge dependent on the given inflowing catchment area. 
    Optionally, the user maydefine inflow points using element IDs and asigning 
    discharges like HQ2 resectively. These will be added at the defined points and
    downstream in the model structure.

    JM 2021

    Arguments:
    -----------
    ser_area_infl: pandas.Series [km2]
        Series of model elements' catchment inflow area corresponding to
        the serie's ascending index. For headwater cells, the value should be zero.
        (e.g., pd.Series([.0, .0, .1, .1], index=[1, 2, 3, 4], name='area_infl'))
    ser_tgb_down: pandas.Series
        Series of corresponding downstream model element indices.
        Model outlet remains -1 and dummy elements are represented as 0.
    q_spec: float [m3s-1km-2] 
        Specific discharge of the catchment for the selected HQ value
    ser_q_in_corr: pandas.Series
        Series of channel-forming inflow (e.g., HQ2) at the corresponding 
        model element ID in the serie's index. 
        (e.g., pd.Series(np.array([2.8, 5.3]), index=[23, 359], name='q_in'))

    Returns:
    -----------
    ser_ch_form_q: pandas.Series
        Series of elements' channel-forming discharge at the corresponding
        model element ID in the serie's index. 
    """    
    
    # calculate channel-forming discharge in [m³/s]
    ser_ch_form_q = ser_area_infl * q_spec
    ser_ch_form_q.name = 'ch_form_q'
    # if inflow series exists, calculate correction values of discharge
    if np.any(ser_q_in_corr):
        # pre-allocate Series
        ser_ch_form_q_corr = pd.Series(np.zeros((ser_ch_form_q.index.shape)),
                                     index=ser_ch_form_q.index, name='corr')
        # iterate inflow points
        for tgb, q_in in ser_q_in_corr.iteritems():
            # calculate inflow to all cells downstream inflow cell
            while tgb != np.max(ser_ch_form_q.index):
                ser_ch_form_q_corr.at[tgb] = ser_ch_form_q_corr.loc[tgb] + q_in
                tgb = ser_tgb_down.at[tgb]
        # correct Q values by inflow
        ser_ch_form_q = ser_ch_form_q + ser_ch_form_q_corr
    
    return ser_ch_form_q

# %% calculate tripel trapezoid river cross section
def calc_ttp(ser_ch_form_q, J_type_routing, ch_est_method='combined',
             def_bx=0, def_bbx_fac=1, def_bnm=1.5, def_bnx=100, def_bnvrx=4, 
             def_skm=30, def_skx=20, print_out=False):
    """
    This function calculates the standard channel cross section triple trapezoid
    profile parameters using estimation functions. Channel estimation functions
    as Allen et al.(1994) depend on onnatural circumstances.
    In many cases, anthropogenic influences are large and often channels are 
    paved or immobilized with stones defining a much larger capacity than naturally
    possible. Therefore Krauter (2006) tries to fit the results of Allen et al. (1994)
    to larger, potentially reshaped rivers. Nevertheless, the function is not
    applicable to very small catchments, as it does not allow infinite small
    channel widths and depths.
    Within this function, any of the named functions may be used, as well as a 
    'combined' version, which selects Allen et al. (1994) for small and Krauter (2006)
    for larger inflow catchment areas.

    JM 2021

    Arguments:
    -----------
    ser_ch_form_q: pandas.Series
        Series of elements' channel-forming discharge at the corresponding
        model element ID in the serie's index. 
    J_type_routing: pandas.Series
        Boolean Series, which identifies the routing cells corresponding to the
        serie's ascending index with True.
        (e.g., pd.Series(data=[0, 0, 1, 1], index=[1, 2, 3, 4],
                         name='routing', dtype='bool'))
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
    print_out: boolean (optional, default: '')
        true if workprogress shall be print to command line

    Returns:
    -----------
    df_ttp: pandas.DataFrame
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
        - 'skm': Strickler roughnes values in the channel [m1/3s-1]
        - 'skl': Strickler roughnes values at the left foreland [m1/3s-1]
        - 'skr': Strickler roughnes values at the right foreland [m1/3s-1]
    """
    if print_out: print('...calculate cross section parameters...')
    # pre-define DataFrane
    df_ttp = pd.DataFrame(np.zeros((ser_ch_form_q.shape[0], 14)),
                          index=ser_ch_form_q.index, 
                          columns=['hm', 'bm', 'bl', 'br', 'bbl', 'bbr',
                                   'bnm', 'bnl', 'bnr', 'bnvrl', 'bnvrr',
                                   'skm', 'skl', 'skr'])
    # using Allen et al. (1994)
    if ch_est_method == 'Allen':
        df_ttp.hm = 0.349 * ser_ch_form_q ** 0.341
        df_ttp.bm = 2.710 * ser_ch_form_q ** 0.557
    # using Krauter (2006)
    elif ch_est_method == 'Krauter':
        df_ttp.hm = 0.328 + 0.028 * (ser_ch_form_q ** (0.388 + 0.022))
        df_ttp.bm = 1.696 + 0.166 * (ser_ch_form_q ** (0.552 + 0.026))
    # combining Allen at al. (1994) for small catchment sizes and Krauter (2006)
    # for others (thresholds: 1.0665 m3s-1 for channel depth and 0.4820 m3s-1
    # for channel width according to the intersection points of Allen and Krauter
    # functions)
    elif ch_est_method == 'combined':
        df_ttp.at[ser_ch_form_q <= 1.0665, 'hm'] \
            = 0.349 * ser_ch_form_q[ser_ch_form_q <= 1.0665] ** 0.341
        df_ttp.at[ser_ch_form_q > 1.0665, 'hm'] = 0.328 \
            + 0.028 * (ser_ch_form_q[ser_ch_form_q > 1.0665] ** (0.388 + 0.022))
        df_ttp.at[ser_ch_form_q <= 0.4820, 'bm'] \
            = 2.710 * ser_ch_form_q[ser_ch_form_q <= 0.4820] ** 0.557
        df_ttp.at[ser_ch_form_q > 0.4820, 'bm'] = 1.696 \
            + 0.166 * (ser_ch_form_q[ser_ch_form_q > 0.4820] ** (0.552 + 0.026))
    # calculate left (BL) and right (BR) flat foreland width
    df_ttp.bl = def_bx
    df_ttp.br = def_bx
    # calculate left (BBL) and right (BBR) slopy foreland width
    df_ttp.bbl = df_ttp.bm * def_bbx_fac
    df_ttp.bbr = df_ttp.bm * def_bbx_fac
    # calculate channel embankment slope (left and right, BNM)
    df_ttp.bnm = def_bnm
    # calculate slopy foreland slope left (BNL) and right (BNR)
    df_ttp.bnl = def_bnx
    df_ttp.bnr = def_bnx
    # calculate outer foreland slope left (BNVRL) and right (BNVRR)
    df_ttp.bnvrl = def_bnvrx
    df_ttp.bnvrr = def_bnvrx
    # calculate Strickler roughness values in the channel (SKM)
    df_ttp.skm = def_skm
    # calculate Strickler roughness at the left (SKL) and right (SKR) foreland
    df_ttp.skl = def_skx
    df_ttp.skr = def_skx
    # clear values for all elements, which are not routing elements
    df_ttp.at[~J_type_routing, :] = np.nan
    return df_ttp

# %% estimate channel water surface level 
def est_ch_wsl(ser_cs_l_m, ser_cs_h_m, ser_tgb_q_in,
               def_cs_hmax_eval=10, def_lam_hres=0.1,
               def_ch_vmin=0.5, def_ch_vmax=3.0):
    """
    This function estimates the water surface level of the channel 
    
    Arguments:
    -----------
    ser_cs_l_m, ser_cs_h_m: pandas.Series
        Series of mean cross sections' distance (ser_cs_l_m) and elevation
        difference (ser_cs_h_m) from flow accumulation's flow paths.
    ser_tgb_q_in: pandas.Series
        Series of elements' channel-forming discharge at the corresponding
        model element ID in the serie's index. 
    def_cs_hmax_eval: float (optional, default: 10) [m]
        maximum height of cross section evaluation
    def_lam_hres: float (optional, default: 0.1) [m]
        spacing between evaluation lamellae
    def_ch_vmin: float (optional, default: 0.5) [m/s]
        minimum reasonable flow velocity
    def_ch_vmax: float (optional, default: 3.0) [m/s]
        maximum reasonable flow velocity
    
    Returns:
    -----------
    h_ll: np.array (int)
        Numpy array of lamellae used to describe the profiles
    ser_wsll_ll, ser_wslm_ll, ser_wslr_ll: pandas.Series
        Series containing the left, mean, and right water surface levels
        for each lamella defined.
    df_ch_h: pandas.DataFrame
        DataFrame containing expected minimum ('min') and maximum ('max')
        water surface levels based on Gaukler-Manning-Strickler equation.
    ser_a_ll, ser_p_ll: pandas.Series
        Series containing the cross section area and wetted perimeter 
        for each lamella defined.
    """
    # pre-define necessary functions
    def insert_lamellae(pts, def_hmax, def_dh):
        """
        This function inserts lamellae intersection points defined with
        maximum lamella elevation and spacing in point list.
        
        Arguments:
        -----------
        pts: list
            original point list
        def_hmax: int
            height of uppermost lamella to add [m]
        def_dh: int
            spacing between lamellae [m]
        
        Returns:
        -----------
        pts_ext: list
            extended point list including lamellae
        """
        
        # pre-allocate matrix
        pts_ext = np.empty((pts.shape[0] + int(def_hmax / def_dh), 4))
        pts_ext.fill(np.nan)
        # initialize iteration over cross section points
        ip = 0
        il_max = int(def_hmax / def_dh)
        for il in range(1, il_max + 1):
            # calculate height of lamella
            Hl = il * def_dh
            # iterate as long as point is lower than lamella
            while pts[ip, 3] < Hl:
                # copy point to new point list
                pts_ext[ip+il-1, :] = pts[ip, :]
                # increase index value
                ip = ip+1
                # if end of point list is reached: break iteration
                if ip > pts.shape[0] - 1:
                    break
            # if end of point list is reached: break iteration
            if ip > pts.shape[0] - 1:
                break
            # get lengths of points for interpolation
            Lps = pts[ip-1:ip+1, 2]
            # get heights of points for interpolation
            Hps = pts[ip-1:ip+1, 3]
            # interpolate lamella length with point information
            Ll = np.interp(Hl, Hps, Lps)
            # insert lamella intersection point in new point list
            pts_ext[ip+il-1, 2:4] = np.hstack((Ll, Hl))
        Ll = pts[ip-1, 2]
        # add lamella points until lamella index equals il_max
        while il <= il_max:
            # calculate height of lamella
            Hl = il * def_dh
            # add lamella intersection point in new point list
            pts_ext[ip+il-1, 2:4] = np.hstack((Ll, Hl))
            # increase index value
            il = il + 1
        # remove empty rows resulting from points higher than il_max
        pts_ext = pts_ext[~np.isnan(pts_ext[:, 3]), :]
        # return result
        return pts_ext
    
    # calculate lamellae
    h_ll = np.arange(def_lam_hres, def_cs_hmax_eval + def_lam_hres, def_lam_hres)
    # pre-define series for interpolated cross section points
    df_ch_h = pd.DataFrame(np.zeros((ser_cs_l_m.shape[0], 2)) * np.nan,
                           index=ser_cs_l_m.index, columns=['min', 'max'])
    ser_wsll_ll = pd.Series(ser_cs_l_m.shape[0]*[[]],
                            index=ser_cs_l_m.index, name='wsll_ll')
    ser_wslm_ll = pd.Series(ser_cs_l_m.shape[0]*[[]],
                            index=ser_cs_l_m.index, name='wslm_ll')
    ser_wslr_ll = pd.Series(ser_cs_l_m.shape[0]*[[]],
                            index=ser_cs_l_m.index, name='wslr_ll')
    ser_a_ll = pd.Series(ser_cs_l_m.shape[0]*[[]],
                         index=ser_cs_l_m.index, name='a_ll')
    ser_p_ll = pd.Series(ser_cs_l_m.shape[0]*[[]],
                         index=ser_cs_l_m.index, name='p_ll')
    # iterate model elements' mean cross sections
    for tgb, (cs_l_m_jj, cs_h_m_jj) \
        in pd.concat((ser_cs_l_m, ser_cs_h_m), axis=1).iterrows():
        # select estimated channel part of cross section
        with np.errstate(invalid='ignore'):
            ch_cs_h = cs_h_m_jj[cs_h_m_jj <= def_cs_hmax_eval]
            ch_cs_l = cs_l_m_jj[cs_h_m_jj <= def_cs_hmax_eval]
                    
        # calculate distances from low point
        # stack signed Euklidian vertex differences to point list
        p_id = np.arange(1, ch_cs_h.shape[0]+1)
        ch_csp = np.transpose(np.vstack((p_id, ch_cs_h, ch_cs_l)))
        # separate left and right values
        ch_cspl = ch_csp[ch_csp[:, 2] <= 0, :]
        ch_cspr = ch_csp[ch_csp[:, 2] >  0, :]
        # sort left part descending
        ch_cspl = np.hstack((
                ch_cspl,
                np.reshape(ch_cspl[:, 1], (ch_cspl.shape[0], 1))))
        ch_cspl = ch_cspl[np.argsort(-ch_cspl[:, 2]), :]
        # sort right part ascending
        ch_cspr = np.hstack((
                ch_cspr,
                np.reshape(ch_cspr[:, 1], (ch_cspr.shape[0], 1))))
        ch_cspr = ch_cspr[np.argsort( ch_cspr[:, 2]), :]
        # add zero point
        ch_cspl = np.vstack((np.zeros((1, 4)), ch_cspl))
        ch_cspr = np.vstack((np.zeros((1, 4)), ch_cspr))
        
        # calculate height differences and
        # insert lamellae intersection points in point lists
        pts_l = ch_cspl[ch_cspl[:, 3] < def_cs_hmax_eval, :]
        ch_csp_extl = insert_lamellae(pts_l, def_cs_hmax_eval, def_lam_hres)
        pts_r = ch_cspr[ch_cspr[:, 3] < def_cs_hmax_eval, :]
        ch_csp_extr = insert_lamellae(pts_r, def_cs_hmax_eval, def_lam_hres)
        
        # calculate area traversed by flow for all points
        # calculate cumulative trapezoidal numerical integration (vert. lamellae)
        ch_ct_al = np.hstack((0, cumtrapz(
                ch_csp_extl[:, 3], abs(ch_csp_extl[:, 2]))))
        ch_ct_ar = np.hstack((0, cumtrapz(
                ch_csp_extr[:, 3], abs(ch_csp_extr[:, 2]))))
        # find indexes of lamellae
        lll = np.isnan(ch_csp_extl[:, 0])
        llr = np.isnan(ch_csp_extr[:, 0])
        # calculate rectangular areas
        ch_r_al = np.multiply(abs(ch_csp_extl[lll, 2]), ch_csp_extl[lll, 3])
        ch_r_ar = np.multiply(abs(ch_csp_extr[llr, 2]), ch_csp_extr[llr, 3])
        # calculate area traversed by flow for all points
        ll_a = (ch_r_al - ch_ct_al[lll]) + (ch_r_ar - ch_ct_ar[llr])
        ser_a_ll.at[tgb] = ll_a
        
        # calculate wetted hydraulic perimeter
        ch_csp_ul = np.cumsum(np.sqrt(
                abs(ch_csp_extl[1:, 2] - ch_csp_extl[:-1, 2]) ** 2 +
                abs(ch_csp_extl[1:, 3] - ch_csp_extl[:-1, 3]) ** 2))
        ch_csp_ur = np.cumsum(np.sqrt(
                abs(ch_csp_extr[1:, 2] - ch_csp_extr[:-1, 2]) ** 2 +
                abs(ch_csp_extr[1:, 3] - ch_csp_extr[:-1, 3]) ** 2))
        ser_p_ll.at[tgb] = ch_csp_ul[lll[1:]] + ch_csp_ur[llr[1:]]
        
        # calculate width of water level
        ser_wsll_ll.at[tgb] = abs(ch_csp_extl[lll, 2])
        ser_wslr_ll.at[tgb] = abs(ch_csp_extr[llr, 2])
        ser_wslm_ll.at[tgb] = abs(ch_csp_extl[lll, 2]) + abs(ch_csp_extr[llr, 2])
        
        # calculate Gaukler-Manning-Strickler velocity and discharge
        # calculate minimum and maximum reasonable discharge
        ll_q_min = def_ch_vmax * ll_a
        ll_q_max = def_ch_vmin * ll_a
        # calculate valid height 
        df_ch_h.at[tgb, 'min'] = griddata(
                ll_q_min, h_ll, ser_tgb_q_in.at[tgb], 'linear')
        df_ch_h.at[tgb, 'max'] = griddata(
                ll_q_max, h_ll, ser_tgb_q_in.at[tgb], 'linear')
    
    return h_ll, ser_wsll_ll, ser_wslm_ll, ser_wslr_ll, df_ch_h, ser_a_ll, ser_p_ll

# %% fit channel depth and width
def fit_ch(ser_pef_hm, ser_pef_bm, ser_cs_l_m, ser_cs_h_m, 
           ser_wsll_ll, ser_wslm_ll, ser_wslr_ll, 
           def_cs_hmax_eval=10,
           def_ch_wmax_eval=40, def_lam_hres=0.1,
           def_chbank_slmin=0.1, def_ch_hmin=0.2,
           def_ch_hmin_eval=0.1, 
           ctrl_show_plots=False, ctrl_save_plots=False, 
           ser_tgb_a=None, ser_tgb_a_in=None, ser_tgb_q_in=None, 
           def_ch_hmax_eval=None, path_plots_out=None):
    
    """
    This function fits the channel bankful water depth and width.
    
    Arguments:
    -----------
    ser_pef_bm, ser_pef_hm: pandas.Series
        Series of estimated channel width ('bm') and depth ('wm')
    ser_cs_l_m, ser_cs_h_m: pandas.Series
        Series of mean cross sections' distance (ser_cs_l_m) and elevation
        difference (ser_cs_h_m) from flow accumulation's flow paths.
    ser_wsll_ll, ser_wslm_ll, ser_wslr_ll: pandas.Series
        Series containing the left, mean, and right water surface levels
        for each lamella defined.
    def_cs_hmax_eval: float (optional, default: 10) [m]
        maximum height of cross section evaluation
    def_ch_wmax_eval: float (optional, default: 40) [m]
        maximum width of channel evaluation
    def_lam_hres: float (optional, default: 0.1) [m]
        spacing between evaluation lamellae
    def_chbank_slmin: float (optional, default: 0.1) [dH/dL]
        minimum riverbank slope threshold for channel identification
    def_ch_hmin: float (optional, default: 0.2, must be >= 0.2) [m]
        minimum channel depth threshold for channel identification
    def_ch_hmin_eval: float (optional, default: 0.1) [m]
        minimum height of channel evaluation
    ctrl_show_plots: boolean (optional, default: False) [-]
        (de-)activate pop-up of figures
    ctrl_save_plots: boolean (optional, default: False) [-]
        (de-)activate export of figures as files
    
    The following arguments are only required for plotting (if either
    ctrl_show_plots or ctrl_save_plots or both is/are True):
    
    ser_tgb_a: pandas.Series (optional, default: None) [km²]
        model element subcatchment area
    ser_tgb_a_in: pandas.Series (optional, default: None) [km²]
        sum of upstream model elements' area
    ser_tgb_q_in: pandas.Series (optional, default: None) [m³/s]
        sum of upstream model elements' river-forming discharge
    def_ch_hmax_eval: float (optional, default: None)
        maximum height of channel evaluation (used to limit y-axis of plot)
    path_plots_out: str (optional, default: None)
        path where plots are stored (e.g., 'c:\model_creation\fig')
    
    Returns:
    -----------
    df_ch_wsl_fit: pandas.DataFrame
        DataFrame containing the 'left', 'mean', and 'right' water surface
        levels for each model element found during the fitting process.
    ser_ch_h_fit: pandas.Series
        Series containing the channel depths for each model element found
        during the fitting process.
    ser_ll_ii_fit: pandas.Series (int)
        Series containing the lamella index for each model element found
        during the fitting process.
    """
    # input error handling section
    if def_ch_hmin < 0.2: sys.exit('def_ch_hmin has to be >= 0.2')
    # pre-define variables
    df_ch_wsl_fit = pd.DataFrame(np.zeros((ser_cs_l_m.shape[0], 3)) * np.nan,
                                 index=ser_cs_l_m.index,
                                 columns=['left', 'mean', 'right'])
    ser_ch_h_fit = pd.Series(np.zeros((ser_cs_l_m.shape[0],)) * np.nan,
                             index=ser_cs_l_m.index, name='h')
    ser_ll_ii_fit = pd.Series(np.zeros((ser_cs_l_m.shape[0], )) - 1,
                              index=ser_cs_l_m.index, name='ll_ii').astype(np.int)
    # calculate lamellae
    h_ll = np.arange(def_lam_hres, def_cs_hmax_eval + def_lam_hres, def_lam_hres)
    # iterate model elements' mean cross sections
    for tgb, (cs_l_m_jj, cs_h_m_jj) \
        in pd.concat((ser_cs_l_m, ser_cs_h_m), axis=1).iterrows():
            # if a mean cross section exists
            if cs_h_m_jj.shape[0] > 0:
                # get cross section part, that certainly includes the channel section
                ch_cs_h = cs_h_m_jj[np.logical_and(
                                cs_l_m_jj >= -def_ch_wmax_eval / 2,
                                cs_l_m_jj <=  def_ch_wmax_eval / 2)]
                ch_cs_l = cs_l_m_jj[np.logical_and(
                                cs_l_m_jj >= -def_ch_wmax_eval / 2,
                                cs_l_m_jj <=  def_ch_wmax_eval / 2)]
                # define histogram bins
                h_hist_bins = np.arange(0, np.ceil(np.nanmax(ch_cs_h)), def_lam_hres)
                # calculate a histogram from cross section elevations
                h_hist, _ = np.histogram(ch_cs_h, bins=h_hist_bins, density=True)
                # find peak of histogram (= foreland elevation)
                h_hist_pk_ii, _ = find_peaks(h_hist, height=def_lam_hres * 10,
                                             prominence=None, plateau_size=(0, 1))
                # only take into account peaks equal or larger threshold
                h_hist_pk_ii = h_hist_pk_ii[h_hist_bins[h_hist_pk_ii] >= def_ch_hmin_eval]
                # if a peak has been identified, save selected lamella
                if h_hist_pk_ii.shape[0] > 0:
                    # if more than one peak has been identified, select the 
                    # closest to the flow network intersection
                    if h_hist_pk_ii.shape[0] > 1:
                        h_hist_pk_ii = h_hist_pk_ii[0]
                    # get center of histogram bins
                    h_hist_bins_c = np.mean(np.vstack((h_hist_bins[:-1],
                                                       h_hist_bins[1: ])), 0)
                    # get bankful water depth
                    ch_cs_h_fit = h_hist_bins_c[h_hist_pk_ii]
                    # get lamella index of lamella right below the bankful 
                    # water depth and lower than the maximum elevation of the
                    # left and right maximum elevation value
                    ll_ii = np.argmin(np.abs(h_ll - ch_cs_h_fit))
                    ch_cs_h_maxl = np.max(ch_cs_h[ch_cs_l < 0])
                    ch_cs_h_maxr = np.max(ch_cs_h[ch_cs_l > 0])
                    while h_ll[ll_ii] > np.min([ch_cs_h_fit, ch_cs_h_maxl, ch_cs_h_maxr]):
                        if ll_ii == 0: break
                        ll_ii -= 1
                    # calculate gradient of channel bank
                    ll_sel_wsl_dif = ser_wslm_ll.at[tgb][ll_ii] \
                        - ser_wslm_ll.at[tgb][ll_ii-1]
                    ll_sel_h_dif = h_ll[ll_ii] - h_ll[ll_ii-1]
                    ll_sel_grad = ll_sel_h_dif / ll_sel_wsl_dif
                    # if gradient of channel bank is smaller slope threshold and 
                    # selected depth is larger than minimum depth threshold
                    # than choose one lamella lower
                    if ll_sel_grad < def_chbank_slmin \
                        and h_ll[ll_ii] > def_ch_hmin:
                        ll_ii -= 1
                        if h_hist_pk_ii > 1:
                            h_hist_pk_ii -= 1
                    # save resulting values
                    ser_ll_ii_fit.at[tgb] = ll_ii
                    ser_ch_h_fit.at[tgb] = h_hist_bins_c[h_hist_pk_ii]
                    df_ch_wsl_fit.at[tgb, 'mean' ] = ser_wslm_ll.at[tgb][ll_ii]
                    df_ch_wsl_fit.at[tgb, 'left' ] = ser_wsll_ll.at[tgb][ll_ii]
                    df_ch_wsl_fit.at[tgb, 'right'] = ser_wslr_ll.at[tgb][ll_ii]
                # if no peak has been identified, use estimation value
                else:
                    ll_ii = np.argmin(np.abs(h_ll - ser_pef_hm.at[tgb]))
                    while h_ll[ll_ii] > ser_pef_hm.at[tgb]:
                        if ll_ii == 0: break
                        ll_ii -= 1
                    ser_ll_ii_fit.at[tgb] = ll_ii
                    ser_ch_h_fit.at[tgb] = ser_pef_hm.at[tgb]
                    df_ch_wsl_fit.at[tgb, 'mean' ] = ser_pef_bm.at[tgb]
                    df_ch_wsl_fit.at[tgb, 'left' ] = ser_pef_bm.at[tgb] / 2
                    df_ch_wsl_fit.at[tgb, 'right'] = ser_pef_bm.at[tgb] / 2
                # create plot including cross section profile, elevation
                # histogram, chosen floodplain elevation, estimated channel
                # depth, and reasonable velocity band width
                if ctrl_show_plots or ctrl_save_plots:
                    # turn plot visibility on or off
                    if ctrl_show_plots: plt.ion()
                    else:               plt.ioff()
                    # create figure and subplot axis handle
                    l_sz = 14
                    fig = plt.figure(figsize=[8, 5])
                    ax = fig.add_subplot(211, position=[.11, .11, .54, .83])
                    # plot mean cross section profile
                    ax.plot(ch_cs_l, ch_cs_h, color='k')
                    # plot fit channel bankful water level
                    ax.plot([-df_ch_wsl_fit.at[tgb, 'left'],
                              df_ch_wsl_fit.at[tgb, 'right']],
                            np.repeat(ser_ch_h_fit.at[tgb], 2), color='b')
                    plt_str = ('model element: {0:d}\nA element: {1:.2f} km²\n'
                               'A inflow: {2:.1f} km²\nQ inflow: {3:.1f} m³/s').format(
                                       tgb, ser_tgb_a.at[tgb],
                                       ser_tgb_a_in.at[tgb], ser_tgb_q_in.at[tgb])
                    plt.legend(['cross section elevation', 'fit bankful water level'],
                               loc='lower right')
                    ax.text(-def_ch_wmax_eval / 2 * 0.95, -0.4, plt_str,
                            fontsize=l_sz - 2, ha='left', va='baseline')
                    # set axis limits
                    ax.set_ylim(-0.5,             def_ch_hmax_eval)
                    ax.set_xlim(-def_ch_wmax_eval / 2, def_ch_wmax_eval / 2 )
                    plt.ylabel('height [m]', fontsize=l_sz)
                    plt.xlabel('width [m]', fontsize=l_sz)
                    plt.xticks(fontsize=l_sz)
                    plt.yticks(fontsize=l_sz)
                    plt.title('channel cross section', fontsize=l_sz)
                    # plot histogram of mean cross section's elevation values
                    ax2 = fig.add_subplot(212, position=[.69, .11, .28, .83])
                    ax2.hist(ch_cs_h, h_hist_bins_c, orientation='horizontal', density=True)
                    # set axis limits
                    ax2.set_ylim(-0.5,             def_ch_hmax_eval)
                    ax2.set_yticklabels([])
                    ax2.set_xlim(0, 1)
                    # set axis lables and title
                    plt.xticks(fontsize=l_sz)
                    plt.yticks(fontsize=l_sz)
                    plt.xlabel('density [-]', fontsize=l_sz)
                    plt.title('elevation histogram', fontsize=l_sz)
                    # show figure, if activated
                    if ctrl_show_plots:
                        plt.show()
                    # save figure, if activated
                    if ctrl_save_plots:
                        # create folder if it does not exist
                        if not os.path.isdir(path_plots_out):
                            os.mkdir(path_plots_out)
                        # save figure
                        plt.savefig('{0:s}ch_h_est_tgb-{1:03d}.png'.format(
                                path_plots_out, int(tgb)), dpi=300)
                    # close figure
                    plt.close(fig)
                    
    return ser_ch_h_fit, df_ch_wsl_fit, ser_ll_ii_fit

# %% write profile.dat file
def write_profdat(df_profdat_par, ser_tgb_csl, path_profdat, 
                  def_cs_hmax_eval, def_lam_hres, 
                  def_profdat_nodata_val=-1, def_profdat_exit_val=999999, 
                  src_geodata='', catch_name='', comment='', 
                  print_out=False):
    r"""
    This function writes a pandas.DataFrame with all necessary parameters to
    the spatial LARSIM input file profile.dat. 

    JM 2021

    Arguments:
    -----------
    df_profdat_par: pandas.DataFrame
        DataFrame of all parameters, which are needed in the resulting file.
        The DataFrame has to include the following columns:
        - ???
        
    ser_tgb_csl: pandas.Series
        Series of cross section ID numbers, which are allocated to all 
        routing model elements in the model structure        
    path_profdat: str
        path including file name, where the file profile.dat shall be stored
        (e.g., 'c:\folder\profile.dat')
    def_profdat_nodata_val: integer (optional, default: -1)
        three character value representing nodata (or NaN) in the resulting file [-]
    def_profdat_exit_val: integer (optional, default: 999999)
        value representing a cross section line block determination [-]
    src_geodata: str (optional, default: '')
        string describing the source of the used geodata
    catch_name: str (optional, default: '')
        name of the catchment, that the model is for
    comment: str (optional, default: '')
        additional text, that shall describe something in the model
    print_out: boolean (optional, default: '')
        true if workprogress shall be print to command line
    
    Returns:
    -----------
    Saves the cross section profile W-A-Q-relation file profile.dat
    """
    if print_out: print('...write cross section profile W-A-Q-relation file...')
    # define internal field names
    f_csl_fid = 'csl_fid'
    # define number of lamellae
    def_l15 = np.int(np.round(def_cs_hmax_eval / def_lam_hres, 0))
    # calculate lamellae
    h_ll = np.arange(def_lam_hres, def_cs_hmax_eval + def_lam_hres, def_lam_hres)
    w_ll = np.int32(np.round(h_ll / def_lam_hres, 0))
    
    df_profdat_exp = copy.deepcopy(df_profdat_par)
    df_profdat_exp.index = df_profdat_exp.loc[:, f_csl_fid]
    df_profdat_exp_ext = df_profdat_exp.loc[ser_tgb_csl, :]
    df_profdat_exp_ext.index = ser_tgb_csl.index[ser_tgb_csl == df_profdat_exp_ext.index]
    # pre-allocate iterator
    block_nb = 0
    # iterate cross sections
    for tgb, (csl, h, a, p, wsl, ll, a_ll, p_ll, wsl_ll) in df_profdat_exp_ext.iterrows():
        # create start line of cross section block
        startline = np.reshape(np.hstack((tgb, wsl / def_lam_hres, a, p, wsl)), (5, 1))
        # create W-A-Q-function vertices in cross section block
        datalines = np.vstack((np.squeeze(np.matlib.repmat(tgb, def_l15, 1)),
                               w_ll, a_ll, p_ll, wsl_ll))
        # create end line of cross section block
        endline = np.reshape(np.hstack((tgb, def_l15 + 1,
                np.squeeze(np.matlib.repmat(def_profdat_exit_val, 1, 3)))), (5, 1))
        # combine cross section block elements
        if block_nb == 0:
            par_exp_unstr = np.hstack((startline, datalines, endline))
        else:
            par_exp_unstr = np.hstack((par_exp_unstr, startline, datalines, endline))
        # increase iterator
        block_nb += 1
    # transpose array
    par_exp_unstr = np.swapaxes(par_exp_unstr, 0, 1)
    # convert unstructured array to structured array
    arr_profdat_par = np.lib.recfunctions.unstructured_to_structured(
            par_exp_unstr, np.dtype([('TGB', 'int64'), ('Wasserstand', 'float64'),
                                  ('Flaeche', 'float64'), ('Umfang', 'float64'),
                                  ('WSB', 'float64')]))
    
    # define variable fields in point feature class
    fields = ['TGB','Wasserstand','Flaeche','Umfang','WSB']
    print_fmt = pd.DataFrame(np.array([
            [6, 12, 10, 10, 10],
            [0,  1,  2,  2,  2]]).T,
        columns=['digits', 'decimals'], index=fields)
    # define heading comment
    if catch_name: prof_catchm_name = '{0:s}'.format(catch_name)
    else: prof_catchm_name = ''
    prof_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    prof_comment = ('#\r'
                    '# tgb.dat: automated production using TATOO package\r'
                    '# author: J. Mitterer, Chair of Hydrology and RBM, '
                    'Technical University Munich\r'
                    '# geodata: {0:s}\r'
                    '# {1:s}\r'
                    '# time: {2:s}\r'
                    '# comment: {3:s}\r'
                    '#\r').format(src_geodata, prof_catchm_name, prof_timestamp, comment)
    # define LARSIM key words
    tgb_keywords = ''
    # define variable titles
    tgb_var_titles = ''
    dig_fmt = '{0:s}'
    for var, dig in print_fmt.digits.iteritems():
        tgb_var_titles = tgb_var_titles + dig_fmt.format(var.rjust(dig) + ';')
    # summarize header lines
    tgb_header = prof_comment + tgb_keywords + tgb_var_titles[:-1]
    # calculate data formats
    fmt_str = []
    for var, (dig, dec) in print_fmt.iterrows():
        if dec == 0: fmt_str.append('%{0:d}d'.format(dig))
        elif dec < 0: fmt_str.append('%{0:d}s'.format(dig))
        else: fmt_str.append('%{0:d}.{1:d}f'.format(dig, dec))
    # write data to file
    np.savetxt(path_profdat, arr_profdat_par[fields],
               delimiter=';', fmt=fmt_str, newline=';\r',
               header=tgb_header, footer='', comments='')
    # import written text file and replace NaN with -1 
    fid = open(path_profdat, 'r')
    tgb_str = fid.read()
    tgb_str = tgb_str.replace('nan', str(def_profdat_nodata_val).rjust(3))
    fid.close()
    # write corrected file string to same file
    fid = open(path_profdat, 'w')
    fid.write(tgb_str)
    fid.close()
