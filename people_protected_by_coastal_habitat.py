"""Pipeline to calculate people protected by coastal hab."""
import logging
import os

from osgeo import gdal
from osgeo import osr
from pygeoprocessing.geoprocessing import _create_latitude_m2_area_column
import pygeoprocessing
import numpy
import scipy
import taskgraph

gdal.SetCacheMax(2**27)

WORKSPACE_DIR = 'workspace'
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')

for dir_path in [ECOSHARD_DIR, CHURN_DIR, WORKSPACE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    #filename='log.out',
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)
logging.getLogger('pygeoprocessing').setLevel(logging.INFO)

TARGET_PIXEL_SIZE = (0.002777777778, -0.002777777778)

# Note: layers are in ecoshard-root/geobon/cv_layers/2000
# This is old-fashioned code that requires all the layers below to be in workspace/ecoshard

POP_RASTER_PATH = os.path.join(
    ECOSHARD_DIR,
    'total_pop_masked_by_10m_2000_md5_a8be07ed5e2afefe03a40dddff03e5b5.tif')

HAB_LAYERS = {
    'reefs': (
        os.path.join(
            ECOSHARD_DIR,
            'reefs_value_md5_b1d862bc42b52ba86c909453bcf1866c.tif'), 2000.0),
    'mangroves_forest': (
        os.path.join(
            ECOSHARD_DIR,
            'mangroves_forest_value_md5_c7309d791ee715e88eacc9ff64376817.tif'), 2000.1),
    'saltmarsh_wetland': (
        os.path.join(
            ECOSHARD_DIR,
            'saltmarsh_wetland_value_md5_f2488a6a777703b19a0e94fe419d96da.tif'), 1000.0),
    'seagrass': (
        os.path.join(
            ECOSHARD_DIR,
            'seagrass_value_md5_3164c8092495286303bc74e9f90d7e99.tif'), 500.0),
    'shrub': (
        os.path.join(
            ECOSHARD_DIR,
            '2_2000_value_md5_fc455be508bb5d96ca35cc86cd8efda8.tif'), 2000.01),
    'sparse': (
        os.path.join(
            ECOSHARD_DIR,
            '4_500_value_md5_311d14db442bea0764915533fffc89f9.tif'), 500.01),
}

#this was for year esa2015: Note: all ecoshards are in gs://ecoshard-root/ipbes-cv
#POP_RASTER_PATH = os.path.join(
#    ECOSHARD_DIR,
#    'total_pop_masked_by_10m_md5_ef02b7ee48fa100f877e3a1671564be2.tif')
#
#HAB_LAYERS = {
#    'reefs': (
#        os.path.join(
#            ECOSHARD_DIR,
#            'reefs_value_md5_42fc7e5155f57102ad22b4e003deb39a.tif'), 2000.0),
#    'mangroves_forest': (
#        os.path.join(
#            ECOSHARD_DIR,
#            'mangroves_forest_value_md5_d53754de7dd71cc12ab2c93937d900b0.tif'), 2000.1),
#    'saltmarsh_wetland': (
#        os.path.join(
#            ECOSHARD_DIR,
#            'saltmarsh_wetland_value_md5_73c36d6f95cdc6227c79ce258140e452.tif'), 1000.0),
#    'seagrass': (
#        os.path.join(
#            ECOSHARD_DIR,
#            'seagrass_value_md5_aa481f29c036e404184795e78c90afd9.tif'), 500.0),
#    'shrub': (
#        os.path.join(
#            ECOSHARD_DIR,
#            '2_2000_value_md5_3a2650575183a61ac1f2e9b8d7d1da1d.tif'), 2000.01),
#    'sparse': (
#        os.path.join(
#            ECOSHARD_DIR,
#            '4_500_value_md5_09b7566d15ffaab23ce7bd86bafc0ccf.tif'), 500.01),
#}


def create_flat_radial_convolution_mask(
        pixel_size_degree, radius_meters, kernel_filepath):
    """Create a radial mask to sample pixels in convolution filter.

    Parameters:
        pixel_size_degree (float): size of pixel in degrees.
        radius_meters (float): desired size of radial mask in meters.

    Returns:
        A 2D numpy array that can be used in a convolution to aggregate a
        raster while accounting for partial coverage of the circle on the
        edges of the pixel.

    """
    degree_len_0 = 110574  # length at 0 degrees
    degree_len_60 = 111412  # length at 60 degrees
    pixel_size_m = pixel_size_degree * (degree_len_0 + degree_len_60) / 2.0
    pixel_radius = numpy.ceil(radius_meters / pixel_size_m)
    n_pixels = (int(pixel_radius) * 2 + 1)
    sample_pixels = 200
    mask = numpy.ones((sample_pixels * n_pixels, sample_pixels * n_pixels))
    mask[mask.shape[0]//2, mask.shape[0]//2] = 0
    distance_transform = scipy.ndimage.morphology.distance_transform_edt(mask)
    mask = None
    stratified_distance = distance_transform * pixel_size_m / sample_pixels
    distance_transform = None
    in_circle = numpy.where(stratified_distance <= radius_meters, 1.0, 0.0)
    stratified_distance = None
    reshaped = in_circle.reshape(
        in_circle.shape[0] // sample_pixels, sample_pixels,
        in_circle.shape[1] // sample_pixels, sample_pixels)
    kernel_array = numpy.sum(reshaped, axis=(1, 3)) / sample_pixels**2
    normalized_kernel_array = kernel_array / numpy.max(kernel_array)
    LOGGER.debug(normalized_kernel_array)
    reshaped = None

    driver = gdal.GetDriverByName('GTiff')
    kernel_raster = driver.Create(
        kernel_filepath.encode('utf-8'), n_pixels, n_pixels, 1,
        gdal.GDT_Float32, options=[
            'BIGTIFF=IF_SAFER', 'TILED=YES', 'BLOCKXSIZE=256',
            'BLOCKYSIZE=256'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_raster.SetGeoTransform([-180, 1, 0, 90, 0, -1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    kernel_raster.SetProjection(srs.ExportToWkt())
    kernel_band = kernel_raster.GetRasterBand(1)
    kernel_band.SetNoDataValue(-1)
    kernel_band.WriteArray(normalized_kernel_array)


def _union_op(*mask_arrays):
    result = numpy.zeros(mask_arrays[0].shape, dtype=numpy.bool)
    for mask_array in mask_arrays:
        result |= (mask_array > 0) & (~numpy.isclose(mask_array, 0))
    return result


def _sum_rasters_op(*value_arrays):
    result = numpy.full(value_arrays[0].shape, -1, dtype=numpy.float32)
    valid = False
    for value_array in value_arrays:
        valid_mask = value_array > 0
        if valid_mask.any():
            valid = True
            LOGGER.debug(numpy.count_nonzero(valid_mask))
            nodata_mask = (result == -1) & valid_mask
            result[nodata_mask] = 0
            result[valid_mask] += value_array[valid_mask]
            LOGGER.debug(numpy.sum(result[valid_mask]))
    if valid:
        LOGGER.debug(numpy.sum(result))
    return result


def _mask_op(mask_array, value_array):
    result = numpy.full(value_array.shape, -1, dtype=numpy.float32)
    valid_array = (mask_array > 0)
    result[valid_array] = value_array[valid_array].astype(numpy.float32)
    return result


def _sum_raster(raster_path):
    running_sum = 0.0
    for _, block_array in pygeoprocessing.iterblocks((raster_path, 1)):
        running_sum += numpy.sum(block_array[block_array > 0])
    LOGGER.info(f'running_sum is: {running_sum}')
    return running_sum


def _mult_by_scalar_op(value_array, scalar):
    result = numpy.full(value_array.shape, -1, dtype=numpy.float32)
    valid_mask = value_array >= 0
    result[valid_mask] = value_array[valid_mask] * scalar
    return result

def warp_by_area(base_raster_path, target_pixel_size, target_raster_path):
    #create a density raster
    density_raster_path = os.path.join(CHURN_DIR, "density.tif")
    _convert_to_density(base_raster_path, density_raster_path)
    #warp the density raster to the target pixel size
    warp_density_raster_path = os.path.join(CHURN_DIR, "warp_density.tif")
    pygeoprocessing.warp_raster(density_raster_path, target_pixel_size, warp_density_raster_path, "average")
    #convert it back to a count
    _density_to_count(warp_density_raster_path, target_raster_path)
    os.remove(density_raster_path)
    os.remove(warp_density_raster_path)

def _density_to_count(
        base_wgs84_density_raster_path, target_wgs84_count_raster_path):
    """Convert base WGS84 raster path to a per density raster path."""
    base_raster_info = pygeoprocessing.get_raster_info(
        base_wgs84_density_raster_path)
    # xmin, ymin, xmax, ymax
    _, lat_min, _, lat_max = base_raster_info['bounding_box']
    _, n_rows = base_raster_info['raster_size']

    m2_area_col = _create_latitude_m2_area_column(lat_min, lat_max, n_rows)
    nodata = base_raster_info['nodata'][0]

    def _mult_by_area_op(base_array, m2_area_array):
        result = numpy.empty(base_array.shape, dtype=base_array.dtype)
        if nodata is not None:
            valid_mask = ~numpy.isclose(base_array, nodata)
            result[:] = nodata
        else:
            valid_mask = numpy.ones(base_array.shape, dtype=bool)

        result[valid_mask] = (
            base_array[valid_mask] * m2_area_array[valid_mask])
        return result

    pygeoprocessing.raster_calculator(
        [(base_wgs84_density_raster_path, 1), m2_area_col], _mult_by_area_op,
        target_wgs84_count_raster_path, base_raster_info['datatype'],
        nodata)


def _convert_to_density(
        base_wgs84_raster_path, target_wgs84_density_raster_path):
    """Convert base WGS84 raster path to a per density raster path."""
    base_raster_info = pygeoprocessing.get_raster_info(
        base_wgs84_raster_path)
    # xmin, ymin, xmax, ymax
    _, lat_min, _, lat_max = base_raster_info['bounding_box']
    _, n_rows = base_raster_info['raster_size']

    m2_area_col = _create_latitude_m2_area_column(lat_min, lat_max, n_rows)
    nodata = base_raster_info['nodata'][0]

    def _div_by_area_op(base_array, m2_area_array):
        result = numpy.empty(base_array.shape, dtype=base_array.dtype)
        if nodata is not None:
            valid_mask = ~numpy.isclose(base_array, nodata)
            result[:] = nodata
        else:
            valid_mask = numpy.ones(base_array.shape, dtype=bool)

        result[valid_mask] = (
            base_array[valid_mask] / m2_area_array[valid_mask])
        return result

    pygeoprocessing.raster_calculator(
        [(base_wgs84_raster_path, 1), m2_area_col], _div_by_area_op,
        target_wgs84_density_raster_path, base_raster_info['datatype'],
        nodata)


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, 4, 5.0)
    task_graph.add_task()
 
    pop_aligned_raster_path = os.path.join(CHURN_DIR, "pop_aligned.tif")

    hab_warp_task = task_graph.add_task(
        func=warp_by_area,
        args=(
            POP_RASTER_PATH, TARGET_PIXEL_SIZE, pop_aligned_raster_path),
        target_path_list=[pop_aligned_raster_path],
        task_name=f'align and resample {pop_aligned_raster_path}')
    hab_warp_task.join()
    pop_raster_info = pygeoprocessing.get_raster_info(pop_aligned_raster_path)

    hab_coverage_task_list = []
    hab_raster_path_list = []
    hab_warp_task_list = []
    pop_coverage_on_raster_list = []
    hab_pop_coverage_task_list = []
    hab_mask_cover_path_list = []
    for hab_key, (unaligned_hab_raster_path, prot_dist) in HAB_LAYERS.items():
        hab_raster_path = os.path.join(
            CHURN_DIR, '%s_aligned%s' % os.path.splitext(os.path.basename(
                unaligned_hab_raster_path)))
        # align the habitat to the population using max so we get all the
        # high to low resolution pixel coverage
        hab_warp_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(
                unaligned_hab_raster_path, TARGET_PIXEL_SIZE,
                hab_raster_path, 'max'),
            kwargs={'target_bb': pop_raster_info['bounding_box']},
            target_path_list=[hab_raster_path],
            task_name=f'align {unaligned_hab_raster_path}')
        hab_raster_path_list.append((hab_raster_path, 1))
        hab_warp_task_list.append(hab_warp_task)

        pixel_size_degree = TARGET_PIXEL_SIZE[0]
        kernel_raster_path = os.path.join(
            CHURN_DIR, f'{hab_key}_{prot_dist}_kernel.tif')

        # this convolution is a flat disk and picks up partial pixels right
        # on the edges of the circle
        kernel_task = task_graph.add_task(
            func=create_flat_radial_convolution_mask,
            args=(
                pixel_size_degree, prot_dist, kernel_raster_path),
            target_path_list=[kernel_raster_path],
            task_name=f'make kernel for {hab_key}')

        # project habitat coverage out the distance that it should cover
        # the values don't matter here just the coverage
        hab_mask_cover_raster_path = os.path.join(
            CHURN_DIR, f'{hab_key}_coverage.tif')
        hab_coverage_task = task_graph.add_task(
            func=pygeoprocessing.convolve_2d,
            args=(
                (hab_raster_path, 1), (kernel_raster_path, 1),
                hab_mask_cover_raster_path),
            kwargs={
                'mask_nodata': False,
                },
            dependent_task_list=[kernel_task, hab_warp_task],
            target_path_list=[hab_mask_cover_raster_path],
            task_name=f'create hab coverage for {hab_key}')
        hab_coverage_task_list.append(hab_coverage_task)
        hab_mask_cover_path_list.append((hab_mask_cover_raster_path, 1))

        # project population out the distance that habitat protects so we
        # can see where the population will intersect with the habitat
        hab_pop_coverage_raster_path = os.path.join(
            CHURN_DIR, f'{hab_key}_pop_coverage.tif')
        pop_coverage_task = task_graph.add_task(
            func=pygeoprocessing.convolve_2d,
            args=(
                (pop_aligned_raster_path, 1), (kernel_raster_path, 1),
                hab_pop_coverage_raster_path),
            kwargs={'mask_nodata': False},
            dependent_task_list=[kernel_task],
            target_path_list=[hab_pop_coverage_raster_path],
            task_name=f'create pop coverage for {hab_key}')

        # mask projected population to hab to see how much population
        # intersects with habitat, the result will be a hab shaped splotch
        # where each pixel represents the number of people within protective
        # distance
        pop_coverage_on_hab_raster_path = os.path.join(
            WORKSPACE_DIR, f'{hab_key}_pop_on_hab.tif')
        hab_mask_pop_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(hab_raster_path, 1), (hab_pop_coverage_raster_path, 1)],
                _mask_op, pop_coverage_on_hab_raster_path,
                gdal.GDT_Float32, -1),
            dependent_task_list=[hab_warp_task, pop_coverage_task],
            target_path_list=[pop_coverage_on_hab_raster_path],
            task_name=f'mask pop by hab effect layer {hab_key}')
        hab_pop_coverage_task_list.append(hab_mask_pop_task)
        pop_coverage_on_raster_list.append(
            (pop_coverage_on_hab_raster_path, 1))

    # combine all the hab coverages into one big raster for total coverage
    total_hab_mask_raster_path = os.path.join(
        CHURN_DIR, 'total_hab_mask_coverage.tif')
    total_hab_mask_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            hab_mask_cover_path_list, _union_op, total_hab_mask_raster_path,
            gdal.GDT_Byte, 0),
        dependent_task_list=hab_coverage_task_list,
        target_path_list=[total_hab_mask_raster_path],
        task_name='total hab mask coverage')

    # mask the population raster by the total hab coverage, this shows
    # how many total people are protected by any habitat
    affected_pop_raster_path = os.path.join(
        CHURN_DIR, 'affected_population.tif')
    total_affectd_pop_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(total_hab_mask_raster_path, 1), (pop_aligned_raster_path, 1)],
            _mask_op, affected_pop_raster_path, gdal.GDT_Float32, -1),
        dependent_task_list=[total_hab_mask_task],
        target_path_list=[affected_pop_raster_path],
        task_name=f'mask pop by hab effect layer')

    # sum the protected population
    sum_mask_pop_task = task_graph.add_task(
        func=_sum_raster,
        args=(affected_pop_raster_path,),
        dependent_task_list=[total_affectd_pop_task],
        store_result=True,
        task_name=f'sum up {affected_pop_raster_path}')

    # calculate the total number of people protected by each habitat pixel
    # all together
    total_pop_coverage_raster_path = os.path.join(
        CHURN_DIR, 'total_pop_coverage_on_hab.tif')
    total_pop_coverage_mask_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            pop_coverage_on_raster_list, _sum_rasters_op,
            total_pop_coverage_raster_path, gdal.GDT_Float32, -1),
        dependent_task_list=hab_pop_coverage_task_list,
        target_path_list=[total_pop_coverage_raster_path],
        task_name='combined population coverage')

    # sum the protected population
    sum_hab_mask_pop_task = task_graph.add_task(
        func=_sum_raster,
        args=(total_pop_coverage_raster_path,),
        dependent_task_list=[total_pop_coverage_mask_task],
        store_result=True,
        task_name=f'sum up {total_pop_coverage_raster_path}')

    # normalize the total population on habitat by the sum of total people
    # protected / sum of total pop hab mask layer
    norm_total_pop_hab_mask_raster_path = os.path.join(
        CHURN_DIR, 'norm_total_pop_hab_mask_coverage.tif')
    norm_total_pop_hab_mask_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (total_pop_coverage_raster_path, 1),
            (sum_mask_pop_task.get()/sum_hab_mask_pop_task.get(), 'raw')],
            _mult_by_scalar_op, norm_total_pop_hab_mask_raster_path,
            gdal.GDT_Float32, -1),
        dependent_task_list=[total_pop_coverage_mask_task],
        target_path_list=[norm_total_pop_hab_mask_raster_path],
        task_name=f'normalize final pop coverage')

    task_graph.join()
    task_graph.close()
    LOGGER.info('all done')


if __name__ == '__main__':
    main()
