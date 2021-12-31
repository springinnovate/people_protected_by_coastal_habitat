"""Pipeline to calculate people protected by coastal hab."""
import argparse
import logging
import os
import multiprocessing

from osgeo import gdal
from osgeo import osr
from ecoshard.geoprocessing.geoprocessing import _create_latitude_m2_area_column
from ecoshard import geoprocessing
from ecoshard import taskgraph
import numpy
import scipy

gdal.SetCacheMax(2**26)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)
logging.getLogger('ecoshard.geoprocessing').setLevel(logging.INFO)

TARGET_PIXEL_SIZE = (0.002777777778, -0.002777777778)


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
    for _, block_array in geoprocessing.iterblocks((raster_path, 1)):
        running_sum += numpy.sum(block_array[block_array > 0])
    LOGGER.info(f'running_sum is: {running_sum}')
    return running_sum


def _mult_by_scalar_op(value_array, scalar):
    result = numpy.full(value_array.shape, -1, dtype=numpy.float32)
    valid_mask = value_array >= 0
    result[valid_mask] = value_array[valid_mask] * scalar
    return result


def warp_by_area(churn_dir, base_raster_path, target_pixel_size, target_raster_path):
    """Warp a raster but scale values by per-pixel area change."""
    density_raster_path = os.path.join(churn_dir, "density.tif")
    _convert_to_density(base_raster_path, density_raster_path)
    warp_density_raster_path = os.path.join(churn_dir, "warp_density.tif")
    geoprocessing.warp_raster(
        density_raster_path, target_pixel_size, warp_density_raster_path,
        "average")
    _density_to_count(warp_density_raster_path, target_raster_path)
    os.remove(density_raster_path)
    os.remove(warp_density_raster_path)

def _density_to_count(
        base_wgs84_density_raster_path, target_wgs84_count_raster_path):
    """Convert base WGS84 raster path to a per density raster path."""
    base_raster_info = geoprocessing.get_raster_info(
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

    geoprocessing.raster_calculator(
        [(base_wgs84_density_raster_path, 1), m2_area_col], _mult_by_area_op,
        target_wgs84_count_raster_path, base_raster_info['datatype'],
        nodata)


def _convert_to_density(
        base_wgs84_raster_path, target_wgs84_density_raster_path):
    """Convert base WGS84 raster path to a per density raster path."""
    base_raster_info = geoprocessing.get_raster_info(
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

    geoprocessing.raster_calculator(
        [(base_wgs84_raster_path, 1), m2_area_col], _div_by_area_op,
        target_wgs84_density_raster_path, base_raster_info['datatype'],
        nodata)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Global CV analysis')
    parser.add_argument(
        'landcover_file',
        help='Path to file that lists landcover scenarios to run.')
    parser.add_argument(
        '--dasgupta_mode', action='store_true',
        help='Ignore offshore mangrove and saltmarsh')
    parser.add_argument(
        '--population', type=str, required=True,
        help='path to the population raster')
    parser.add_argument(
        '--reefs', type=str, required=True,
        help='path to the reefs raster')
    parser.add_argument(
        '--mangroves_forest', type=str, required=True,
        help='path to the mangroves_forest raster')
    parser.add_argument(
        '--saltmarsh_wetland', type=str, required=True,
        help='path to the saltmarsh_wetland raster')
    parser.add_argument(
        '--seagrass', type=str, required=True,
        help='path to the seagrass raster')
    parser.add_argument(
        '--shrub', type=str, required=True,
        help='path to the shrub raster')
    parser.add_argument(
        '--sparse', type=str, required=True,
        help='path to the sparse raster')
    parser.add_argument(
        '--prefix', type=str, required=True,
        help='path to the output prefix')
    args = parser.parse_args()

    hab_layers = {
        'reefs': (args.reefs, 2000.0),
        'mangroves_forest': (args.mangroves_forest, 2000.1),
        'saltmarsh_wetland': (args.saltmarsh_wetland, 1000.0),
        'seagrass': (args.seagrass, 500.0),
        'shrub': (args.shrub, 2000.01),
        'sparse': (args.sparse, 500.01),
        }

    workspace_dir = f'workspace_{args.prefix}'
    churn_dir = os.path.join(workspace_dir, 'churn')

    for dir_path in [churn_dir, workspace_dir]:
        os.makedirs(dir_path, exist_ok=True)

    task_graph = taskgraph.TaskGraph(
        workspace_dir, multiprocessing.cpu_count(), 15.0)
    task_graph.add_task()

    pop_aligned_raster_path = os.path.join(churn_dir, "pop_aligned.tif")

    hab_warp_task = task_graph.add_task(
        func=warp_by_area,
        args=(
            churn_dir, args.population, TARGET_PIXEL_SIZE,
            pop_aligned_raster_path),
        target_path_list=[pop_aligned_raster_path],
        task_name=f'align and resample {pop_aligned_raster_path}')
    hab_warp_task.join()
    pop_raster_info = geoprocessing.get_raster_info(pop_aligned_raster_path)

    hab_coverage_task_list = []
    hab_raster_path_list = []
    hab_warp_task_list = []
    pop_coverage_on_raster_list = []
    hab_pop_coverage_task_list = []
    hab_mask_cover_path_list = []
    for hab_key, (unaligned_hab_raster_path, prot_dist) in hab_layers.items():
        hab_raster_path = os.path.join(
            churn_dir, '%s_aligned%s' % os.path.splitext(os.path.basename(
                unaligned_hab_raster_path)))
        # align the habitat to the population using max so we get all the
        # high to low resolution pixel coverage
        hab_warp_task = task_graph.add_task(
            func=geoprocessing.warp_raster,
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
            churn_dir, f'{hab_key}_{prot_dist}_kernel.tif')

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
            churn_dir, f'{hab_key}_coverage.tif')
        hab_coverage_task = task_graph.add_task(
            func=geoprocessing.convolve_2d,
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
            churn_dir, f'{hab_key}_pop_coverage.tif')
        pop_coverage_task = task_graph.add_task(
            func=geoprocessing.convolve_2d,
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
            args.workspace_dir, f'{hab_key}_pop_on_hab.tif')
        hab_mask_pop_task = task_graph.add_task(
            func=geoprocessing.raster_calculator,
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
        churn_dir, 'total_hab_mask_coverage.tif')
    total_hab_mask_task = task_graph.add_task(
        func=geoprocessing.raster_calculator,
        args=(
            hab_mask_cover_path_list, _union_op, total_hab_mask_raster_path,
            gdal.GDT_Byte, 0),
        dependent_task_list=hab_coverage_task_list,
        target_path_list=[total_hab_mask_raster_path],
        task_name='total hab mask coverage')

    # mask the population raster by the total hab coverage, this shows
    # how many total people are protected by any habitat
    affected_pop_raster_path = os.path.join(
        churn_dir, 'affected_population.tif')
    total_affectd_pop_task = task_graph.add_task(
        func=geoprocessing.raster_calculator,
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
        churn_dir, 'total_pop_coverage_on_hab.tif')
    total_pop_coverage_mask_task = task_graph.add_task(
        func=geoprocessing.raster_calculator,
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
        churn_dir, 'norm_total_pop_hab_mask_coverage.tif')
    norm_total_pop_hab_mask_task = task_graph.add_task(
        func=geoprocessing.raster_calculator,
        args=([
            (total_pop_coverage_raster_path, 1),
            (sum_mask_pop_task.get()/sum_hab_mask_pop_task.get(), 'raw')],
            _mult_by_scalar_op, norm_total_pop_hab_mask_raster_path,
            gdal.GDT_Float32, -1),
        dependent_task_list=[total_pop_coverage_mask_task],
        target_path_list=[norm_total_pop_hab_mask_raster_path],
        task_name=f'normalize final pop coverage {norm_total_pop_hab_mask_raster_path}')

    task_graph.join()
    task_graph.close()
    LOGGER.info('all done')


if __name__ == '__main__':
    main()
