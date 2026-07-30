from typing import Tuple
import numpy as np
import pandas as pd
import scanpy as sc
import dask.array as da
import argparse
from rasterio.features import rasterize
from rasterio.transform import from_origin
from bin2cell import destripe, process_bins_pd
import skimage
import SimpleITK as sitk
from tifffile import imwrite

# -------------------- CONSTANTS --------------------
CLAHE_CLIP_LIMIT = 0.03
GAUSSIAN_SIGMA = 2
# -------------------- ARGUMENT PARSING --------------------
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description='Postprocess')
    parser.add_argument('-ba', '--bins_adata', type=str, required=True, help='bins_adata path')
    parser.add_argument('-bp', '--bins_parquet', type=str, required=True, help='bins_parquet path')
    parser.add_argument('-sf', '--scale_fac', type=str, required=True, help='bins scale factors')
    parser.add_argument('-hz', '--he_zarr', type=str, required=True, help='path to zarr')
    parser.add_argument('-mpp', '--microns_per_pixel', type=float, required=False, default=0.11, help='scan mpp')
    parser.add_argument('-ilvl', '--image_level', type=int, required=False, default=3, help='level to use for realignment')
    return parser.parse_args()

# -------------------- IMAGE PROCESSING FUNCTIONS --------------------
def extract_hematoxylin_component(he_rgb: np.ndarray) -> np.ndarray:
    """
    Extract the hematoxylin channel from an H&E RGB image.
    Returns the HED image (Hematoxylin-Eosin-DAB).
    """
    he_hed = skimage.color.rgb2hed(he_rgb)
    return he_hed  

def normalize_CLAHE(
    hematoxylin: np.ndarray, 
    raster: np.ndarray, 
    clip_limit: float = CLAHE_CLIP_LIMIT
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize both the hematoxylin and raster images using CLAHE and Gaussian smoothing.
    Returns the normalized fixed (hematoxylin) and moving (raster) images.
    """
    # CLAHE normalization for the fixed image
    fixed = hematoxylin.copy()
    fixed = skimage.exposure.equalize_adapthist(fixed / (fixed.max() + 1e-9), clip_limit=clip_limit)
    # Gaussian smoothing and CLAHE for the moving image
    moving = skimage.filters.gaussian(raster, sigma=GAUSSIAN_SIGMA)
    moving = np.log1p(1 + moving)
    moving = skimage.exposure.equalize_adapthist(moving / (moving.max() + 1e-9), clip_limit=clip_limit)
    return fixed, moving

def process_adata(adata: sc.AnnData, bins: pd.DataFrame, to_destripe=False) -> Tuple[sc.AnnData, pd.DataFrame]:
    """
    Update AnnData and bins with total counts and destripe the data.
    """
    adata.var_names_make_unique()
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    bins.loc[:, 'total_counts'] = bins.set_index('barcode').assign(
        total_counts=adata.obs.total_counts
    ).total_counts.values
    adata.obs = pd.concat([adata.obs, bins.set_index('barcode')], axis=1)
    adata.obs = adata.obs.drop(columns=['total_counts'])
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    if to_destripe:
        destripe(adata, counts_key='total_counts')
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    bins.loc[:, 'total_counts'] = bins.set_index('barcode').assign(
        total_counts=adata.obs.total_counts
    ).total_counts.values
    return adata, bins

def command_iteration(method):
    """Callback for SimpleITK registration optimizer iteration."""
    if method.GetOptimizerIteration() == 0:
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():7.5f} "
        + f": {method.GetOptimizerPosition()}"
    )

def affine_reg_sitk(hematoxylin, raster):
    """
    Perform affine registration using SimpleITK.
    Returns the fixed, moving, and registered images.
    """
    fixed = sitk.GetImageFromArray(hematoxylin)
    moving = sitk.GetImageFromArray(raster)
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
    R.SetOptimizerScalesFromIndexShift()
    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform())
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    outTx = R.Execute(fixed, moving)
    print("-------")
    print(outTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")

    # Resample the moving image to the fixed image space
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)
    out = resampler.Execute(moving)
    return fixed, moving, outTx, out

def sitk_to_affine_params(matrix, translation):
    # Shapely expects [a, b, d, e, xoff, yoff] (affine 2D)
    return [matrix[0,0], matrix[0,1],
            matrix[1,0], matrix[1,1],
            translation[0], translation[1]]

def apply_transform(x: float, y: float, tx: np.ndarray) -> tuple[float, float]:
    """Apply affine transform to x,y coordinates."""
    # tx format: [a, b, d, e, xoff, yoff]
    a, b, d, e, xoff, yoff = tx
    new_x = a * x + b * y + xoff
    new_y = d * x + e * y + yoff
    return new_x, new_y

# -------------------- MAIN FUNCTION --------------------
def main():
    """
    Main workflow for alignment correction:
    - Loads H&E image and extracts hematoxylin
    - Loads bins and AnnData, processes counts
    - Rasterizes bins to image
    - Normalizes and registers images
    - Saves composite OME-TIFF
    - Applies correction to bins.parquet
    """
    args = parse_arguments()

    IMAGE_LEVEL = args.image_level
    PIXEL_SIZE = 2 ** IMAGE_LEVEL
    CAO = 200 # capture area offset pxs
    
    # OME-TIFF metadata for output
    metadata = {
        'axes': 'CZTYX', 
        'PhysicalSizeX': args.microns_per_pixel * PIXEL_SIZE,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': args.microns_per_pixel * PIXEL_SIZE,
        'PhysicalSizeYUnit': 'µm',
        'Description': f'Hematoxylin-counts density overlay, downscaled by factor {PIXEL_SIZE}',
        'Channel': {'Name': ['Hematoxylin', 'Counts', 'Counts_corrected'],
        },
    }

    # Load H&E image from zarr and extract hematoxylin channel
    he = da.from_zarr(f'{args.he_zarr}/0/{IMAGE_LEVEL}/', chunks=(1, 3, 1, 2048, 2048)).squeeze().transpose(1, 2, 0)
    hematoxylin = he.map_blocks(extract_hematoxylin_component, dtype=np.float32)[:, :, 0].compute()

    # Load bins and AnnData, process counts
    bins, _ = process_bins_pd(args.bins_parquet, args.scale_fac)
    adata = sc.read_10x_h5(args.bins_adata)
    adata, bins = process_adata(adata, bins)

    # Prepare rasterization transform
    he_shape = da.from_zarr(f'{args.he_zarr}/0/0/').squeeze().shape
    transform = from_origin(0, he_shape[1], PIXEL_SIZE, PIXEL_SIZE)

    # Rasterize bins to image
    shapes = [(geom, value) for geom, value in zip(bins.geometry, bins["total_counts"])]
    height, width = hematoxylin.shape
    raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype='float32'
    )
    raster = raster[::-1, ]  # Flip along y axis
    raster[np.isinf(raster)] = raster[~np.isinf(raster)].max()
    mask = raster > 0
    props = skimage.measure.regionprops(mask.astype(int))
    minr, minc, maxr, maxc = props[0].bbox
    
    y_len, x_len = hematoxylin.shape
    y0, y1, x0, x1 = max(0,minr-CAO), min(y_len, maxr+CAO), max(0,minc-CAO), min(x_len, maxc+CAO)
    raster = raster[y0:y1, x0:x1]
    hematoxylin = hematoxylin[y0:y1, x0:x1]
    
    # Save raw arrays for debugging
    np.save('bins_raster_raw.npy', raster)
    np.save('hematoxylin_raw.npy', hematoxylin)

    # Normalize both images
    hematoxylin_norm, raster_norm = normalize_CLAHE(hematoxylin, raster)

    # Register images using affine registration
    fixed, moving, outTx, out = affine_reg_sitk(hematoxylin_norm, raster_norm)

    matrix = np.array(outTx.GetMatrix()).reshape(2, 2)
    translation = np.array(outTx.GetTranslation()) * PIXEL_SIZE
    affine_params = sitk_to_affine_params(matrix, translation)
    np.save('transforms.npy', affine_params)

    # Apply correction to bins.parquet
    bins_df = pd.read_parquet(args.bins_parquet)
    new_coords = np.array([
        apply_transform(x, y, affine_params) 
        for x, y in zip(bins_df['pxl_col_in_fullres'], bins_df['pxl_row_in_fullres'])
    ])
    bins_df['pxl_col_in_fullres'] = new_coords[:, 0]
    bins_df['pxl_row_in_fullres'] = new_coords[:, 1]
    bins_df.to_parquet('bins_corrected.parquet')
    
    # Stack and save as OME-TIFF
    arr = np.stack([
        sitk.GetArrayFromImage(fixed),
        sitk.GetArrayFromImage(moving),
        sitk.GetArrayFromImage(out).clip(0, 1)
    ])  # shape: (3, Y, X)
    arr = arr[np.newaxis, ...]
    arr = arr[np.newaxis, ...]
    arr = arr.transpose(2,0,1,3,4)
    arr.shape
    
    imwrite(
        'composite_corrected.tif',
        arr,
        tile=(128, 128),
        compression='deflate',
        photometric='minisblack',
        metadata=metadata,
        ome=True
    )

# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    main()