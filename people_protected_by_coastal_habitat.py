"""Pipeline to calculate people protected by coastal hab."""
import logging
import os

from osgeo import gdal
from osgeo import osr
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


POP_RASTER_PATH = os.path.join(
    ECOSHARD_DIR,
    'total_pop_masked_by_10m_md5_ef02b7ee48fa100f877e3a1671564be2.tif')

HAB_LAYERS = {
    'reefs': (
        os.path.join(
            ECOSHARD_DIR,
            'ipbes-cv_reef_wgs84_compressed_md5_96d95cc4f2c5348394eccff9e8b84e6b.tif'), 2000.0),
    'mangroves_forest': (
        os.path.join(
            ECOSHARD_DIR,
            'ipbes-cv_mangrove_md5_0ec85cb51dab3c9ec3215783268111cc.tif'), 2000.1),
    'saltmarsh_wetland': (
        os.path.join(
            ECOSHARD_DIR,
            'ipbes-cv_saltmarsh_md5_203d8600fd4b6df91f53f66f2a011bcd.tif'), 1000.0),
    'seagrass': (
        os.path.join(
            ECOSHARD_DIR,
            'ipbes-cv_seagrass_md5_a9cc6d922d2e74a14f74b4107c94a0d6.tif'), 500.0),
}


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
    for value_array in value_arrays:
        valid_mask = value_array > 0
        result[valid_mask] += value_array[valid_mask]
    return result


def _mask_op(mask_array, value_array):
    result = numpy.full(value_array.shape, -1, dtype=numpy.float32)
    valid_array = mask_array & (value_array > 0)
    result[valid_array] = value_array[valid_array]
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


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, 4, 5.0)
    task_graph.add_task()

    pop_raster_info = pygeoprocessing.get_raster_info(POP_RASTER_PATH)

    hab_coverage_task_list = []
    hab_raster_path_list = []
    hab_pop_coverage_raster_list = []
    hab_pop_coverage_task_list = []
    for hab_key, (unaligned_hab_raster_path, prot_dist) in HAB_LAYERS.items():
        hab_raster_path = os.path.join(
            CHURN_DIR, '%s_aligned%s' % os.path.splitext(os.path.basename(
                unaligned_hab_raster_path)))
        # align the habitat to the population using max so we get all the
        # high to low resolution pixel coverage
        warp_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(
                unaligned_hab_raster_path, pop_raster_info['pixel_size'],
                hab_raster_path, 'max'),
            kwargs={'target_bb': pop_raster_info['bounding_box']},
            target_path_list=[hab_raster_path],
            task_name=f'align {unaligned_hab_raster_path}')

        pixel_size_degree = pop_raster_info['pixel_size'][0]
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
            dependent_task_list=[kernel_task, warp_task],
            target_path_list=[hab_mask_cover_raster_path],
            task_name=f'create hab coverage for {hab_key}')
        hab_coverage_task_list.append(hab_coverage_task)
        hab_raster_path_list.append((hab_raster_path, 1))

        # project population out the distance that habitat protects so we
        # can see where the population will intersect with the habitat
        hab_pop_coverage_raster_path = os.path.join(
            CHURN_DIR, f'{hab_key}_pop_coverage.tif')
        pop_coverage_task = task_graph.add_task(
            func=pygeoprocessing.convolve_2d,
            args=(
                (POP_RASTER_PATH, 1), (kernel_raster_path, 1),
                hab_pop_coverage_raster_path),
            kwargs={'mask_nodata': False},
            dependent_task_list=[kernel_task],
            target_path_list=[hab_pop_coverage_raster_path],
            task_name=f'create pop coverage for {hab_key}')

        # mask projected population to hab to see how much population
        # intersects with habitat, the result will be a hab shaped splotch
        # where each pixel represents the number of people within protective
        # distance
        hab_pop_coverage_on_hab_raster_path = os.path.join(
            WORKSPACE_DIR, f'{hab_key}_pop_on_hab.tif')
        mask_pop_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(hab_raster_path, 1), (hab_pop_coverage_raster_path, 1)],
                _mask_op, hab_pop_coverage_on_hab_raster_path, gdal.GDT_Float32, -1),
            dependent_task_list=[pop_coverage_task],
            target_path_list=[hab_pop_coverage_on_hab_raster_path],
            task_name=f'mask pop by hab effect layer {hab_key}')
        hab_pop_coverage_task_list.append(pop_coverage_task)
        hab_pop_coverage_raster_list.append(
            (hab_pop_coverage_raster_path, 1))

    # combine all the hab coverages into one big raster for total coverage
    total_hab_mask_raster_path = os.path.join(
        CHURN_DIR, 'total_hab_mask_coverage.tif')
    total_hab_mask_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            hab_raster_path_list, _union_op, total_hab_mask_raster_path,
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
            [(total_hab_mask_raster_path, 1), (POP_RASTER_PATH, 1)],
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
        CHURN_DIR, 'total_pop_coverage.tif')
    total_pop_coverage_mask_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            hab_pop_coverage_raster_list, _sum_rasters_op,
            total_pop_coverage_raster_path, gdal.GDT_Float32, -1),
        dependent_task_list=hab_coverage_task_list,
        target_path_list=[total_pop_coverage_raster_path],
        task_name='combined population coverage')

    total_pop_hab_mask_raster_path = os.path.join(
        CHURN_DIR, 'total_pop_hab_mask_coverage.tif')
    total_pop_hab_mask_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(total_hab_mask_raster_path, 1),
             (total_pop_coverage_raster_path, 1)], _mask_op,
            total_pop_hab_mask_raster_path, gdal.GDT_Float32, -1),
        dependent_task_list=[total_pop_coverage_mask_task],
        target_path_list=[total_pop_hab_mask_raster_path],
        task_name='total pop coverage masked by hab')

    # sum the protected population
    sum_hab_mask_pop_task = task_graph.add_task(
        func=_sum_raster,
        args=(total_pop_hab_mask_raster_path,),
        dependent_task_list=[total_pop_hab_mask_task],
        store_result=True,
        task_name=f'sum up {total_pop_hab_mask_raster_path}')

    # normalize the total population on habitat by the sum of total people
    # protected / sum of total pop hab mask layer
    norm_total_pop_hab_mask_raster_path = os.path.join(
        CHURN_DIR, 'norm_total_pop_hab_mask_coverage.tif')
    norm_total_pop_hab_mask_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (total_pop_hab_mask_raster_path, 1),
            (sum_mask_pop_task.get()/sum_hab_mask_pop_task.get(), 'raw')],
            _mult_by_scalar_op, norm_total_pop_hab_mask_raster_path,
            gdal.GDT_Float32, -1),
        dependent_task_list=[total_pop_hab_mask_task],
        target_path_list=[norm_total_pop_hab_mask_raster_path],
        task_name=f'normalize final pop coverage')

    task_graph.join()
    task_graph.close()
    LOGGER.info('all done')


if __name__ == '__main__':
    main()
