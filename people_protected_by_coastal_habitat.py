"""Pipeline to calculate people protected by coastal hab."""
import logging
import os

from osgeo import gdal
from osgeo import osr
import pygeoprocessing
import numpy
import scipy
import taskgraph

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
logging.getLogger('taskgraph').setLevel(logging.WARN)
logging.getLogger('pygeoprocessing').setLevel(logging.WARN)


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
            'ipbes-cv_mangrove_md5_0ec85cb51dab3c9ec3215783268111cc.tif'), 2000.0),
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
    in_circle = numpy.where(stratified_distance <= 2000.0, 1.0, 0.0)
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


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, 4)
    task_graph.add_task()

    for hab_key, hab_raster_path, prot_dist in HAB_LAYERS.items():
        hab_raster_info = pygeoprocessing.get_raster_info(
            hab_raster_path)
        pixel_size_degree = hab_raster_info['pixel_size'][0]
        kernel_raster_path = os.path.join(
            WORKSPACE_DIR, f'{hab_key}_{prot_dist}_kernel.tif')

        kernel_task = task_graph.add_task(
            func=create_flat_radial_convolution_mask,
            args=(
                pixel_size_degree, prot_dist, kernel_raster_path),
            target_path_list=[kernel_raster_path],
            task_name=f'make kernel for {hab_key}')

        hab_mask_cover_raster_path = os.path.join(
            CHURN_DIR, f'{hab_key}_coverage.tif')
        task_graph.add_task(
            func=pygeoprocessing.convolve_2d,
            args=(
                (hab_raster_path, 1), (kernel_raster_path, 1),
                hab_mask_cover_raster_path),
            kwargs={'mask_nodata': True},
            target_path_list=[hab_mask_cover_raster_path],
            task_name=f'create hab coverage for {hab_key}'

    task_graph.join()
    task_graph.close()
    LOGGER.info('all done')

if __name__ == '__main__':
    main()
