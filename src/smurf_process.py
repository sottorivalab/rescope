import argparse
import json
import pickle
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import copy
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import dask.array as da
import geopandas as gpd
import shapely
from shapely.geometry import Polygon
from rasterio.features import rasterize
from rasterio.transform import from_origin
import torch
from tqdm import tqdm

import src.smurf as su

import os
# Type aliases
PathLike = Union[str, Path]
Device = torch.device

@dataclass
class SmurfConfig:
    """Configuration for SMURF processing."""
    bins_adata: Path
    bins_parquet: Path
    cellvit_json: Path
    zarr_path: Path
    im_path: Path
    output_dir: Path
    transforms: Optional[Path]
    delta_x: float
    delta_y: float
    alignment: Optional[Path]
    device: Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments() -> SmurfConfig:
    """Parse command line arguments into a SmurfConfig object."""
    parser = argparse.ArgumentParser(description='SMURF: Spatial Multi-omics Unsupervised Random Forest')
    
    # Required arguments
    parser.add_argument('-ba', '--bins_adata', type=str, required=True,
                       help='Path to bins AnnData file')
    parser.add_argument('-bp', '--bins_parquet', type=str, required=True,
                       help='Path to bins parquet file')
    parser.add_argument('-cj', '--cellvit_json', type=str, required=True,
                       help='Path to CellViT results JSON file')
    parser.add_argument('-zp', '--zarr_path', type=str, required=True,
                         help='Path to zarr array')
    parser.add_argument('-ip', '--im_path', type=str, required=True,
                         help='Path to image')
    
    # Optional arguments
    parser.add_argument('-od', '--output_dir', type=str, default='./',
                       help='Output path')
    parser.add_argument('-tx', '--transforms', type=str, required=False,
                       help='Path to complex transformations file')
    parser.add_argument('-dx', '--delta_x', type=float, default=0.0,
                       help='X offset for alignment')
    parser.add_argument('-dy', '--delta_y', type=float, default=0.0,
                       help='Y offset for alignment')
    parser.add_argument('-a', '--alignment', type=str, required=False,
                       help='Path to alignment correction JSON file')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    config = SmurfConfig(
        bins_adata=Path(args.bins_adata),
        bins_parquet=Path(args.bins_parquet),
        cellvit_json=Path(args.cellvit_json),
        zarr_path=Path(args.zarr_path),
        im_path=Path(args.im_path),
        output_dir=Path(args.output_dir),
        transforms=Path(args.transforms) if args.transforms else None,
        delta_x=args.delta_x,
        delta_y=args.delta_y,
        alignment=Path(args.alignment) if args.alignment else None,
    )
    
    # Apply alignment correction if provided
    if config.alignment:
        with open(config.alignment, 'r') as f:
            alignment = json.load(f)
            config.delta_x = float(alignment.get('dx', 0.0))
            config.delta_y = float(alignment.get('dy', 0.0))
    
    return config

def save_compressed_pickle(data: any, filename: PathLike) -> None:
    """Save data as a compressed pickle file."""
    with gzip.GzipFile(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def read_cells_outlines(json_path: PathLike) -> gpd.GeoDataFrame:
    """Read cell outlines from CellViT JSON output."""
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {json_path}: {e}")
    
    cell_records = []
    for cell in tqdm(data['cells'], desc="Processing cells"):
        try:
            geometry = Polygon(cell['contour'])
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
        except Exception as e:
            print(f"Failed to process cell contour: {e}")
            continue
        
        cell_record = {
            "geometry": geometry,
            "type": cell["type"],
            "type_label": data['type_map'][str(cell["type"])],
            "type_prob": cell["type_prob"],
            "bbox": cell["bbox"],
            "centroid": cell["centroid"],
            "patch_coordinates": cell["patch_coordinates"],
            "cell_status": cell["cell_status"],
            "offset_global": cell["offset_global"],
            "edge_position": cell["edge_position"]
        }
        cell_records.append(cell_record)
    
    cell_records = gpd.GeoDataFrame(cell_records).set_geometry("geometry")
    cell_records = cell_records.reset_index()
    cell_records = cell_records.assign(cell_id=cell_records['index'].astype(int) + 1)
    return cell_records

def write_labeled_segmentation(so: any, zarr_path: PathLike, cell_records: gpd.GeoDataFrame) -> np.ndarray:
    """Write labeled segmentation to a raster file."""
    row_left = max(so.df_temp["pxl_row_left_in_fullres"].min(), 0)
    row_right = so.df_temp["pxl_row_right_in_fullres"].max()
    col_up = max(so.df_temp["pxl_col_up_in_fullres"].min(), 0)
    col_down = so.df_temp["pxl_col_down_in_fullres"].max()
    
    he_shape = da.from_zarr(f'{zarr_path}/0/0').squeeze().shape
    transform = from_origin(0, he_shape[1], 1, 1)
    shapes = [(geom, value) for geom, value in cell_records[['geometry', 'cell_id']].values]
    
    raster = rasterize(
        shapes=shapes,
        out_shape=(he_shape[1], he_shape[2]),
        transform=transform,
        fill=0,
        dtype='int'
    )
    raster = raster[::-1, ]
    raster = raster[row_left:row_right, col_up:col_down]
    with gzip.GzipFile('segmentation.npy.gz', 'w') as f:
        np.save(f, raster)
    return raster

def process_data(config: SmurfConfig) -> None:
    """Main processing function."""
    # Prepare dataframe and image
    so = su.prepare_dataframe_image(
        config.bins_parquet,
        config.im_path,
        'HE'
    )
    
    # Read and process cell outlines
    cell_records = read_cells_outlines(config.cellvit_json)
    raster = write_labeled_segmentation(so, config.zarr_path, cell_records)
    
    # Generate cell spots information
    so.segmentation_final = raster.copy()
    so.generate_cell_spots_information()
    
    # Read and process AnnData
    adata = sc.read_10x_h5(str(config.bins_adata))
    adata = adata[so.df[so.df.in_tissue == 1]['barcode']].copy()
    sc.pp.filter_genes(adata, min_counts=1000)
    
    # Process nuclei and RNA
    su.nuclei_rna(adata, so)
    adata_sc = copy.deepcopy(so.final_nuclei)
    sc.pp.filter_cells(adata_sc, min_counts=5)
    adata_raw = copy.deepcopy(adata_sc)

    save_path = config.output_dir / 'smurf_intermediate' 
    os.makedirs(save_path, exist_ok=True)
    
    adata_sc = su.singlecellanalysis(adata_sc,resolution=2)
    # Perform iterative arrangement
    su.itering_arragement(
        adata_sc,
        adata_raw,
        adata,
        so,
        resolution=2,
        save_folder=str(save_path)+'/',
        show=True,
        keep_previous=False
    )
    
    # Read final results
    adatas_final = sc.read_h5ad(str(save_path / 'adatas.h5ad'))
    with open(str(save_path / 'cells_final.pkl'), 'rb') as file:
        cells_final = pickle.load(file)
    with open(str(save_path / 'weights_record.pkl'), 'rb') as file:
        weights_record = pickle.load(file)
    
    # Prepare data for optimization
    (pct_toml_dic, spots_X_dic, celltypes_dic, cells_X_plus_dic,
     nonzero_indices_dic, nonzero_indices_toml, cells_before_ml,
     cells_before_ml_x, groups_combined, spots_id_dic,
     spots_id_dic_prop) = su.make_preparation(
         cells_final, so, adatas_final, adata, weights_record,
         maximum_cells=6000
     )
    
    # Save intermediate results
    save_compressed_pickle(spots_X_dic, save_path / "spots_X_dic.pkl.gz")
    save_compressed_pickle(celltypes_dic, save_path / "celltypes_dic.pkl.gz")
    save_compressed_pickle(nonzero_indices_dic, save_path / "nonzero_indices_dic.pkl.gz")
    save_compressed_pickle(nonzero_indices_toml, save_path / "nonzero_indices_toml.pkl.gz")
    save_compressed_pickle(cells_X_plus_dic, save_path / "cells_X_plus_dic.pkl.gz")
    save_compressed_pickle(cells_before_ml, save_path / "cells_before_ml.pkl.gz")
    save_compressed_pickle(cells_before_ml_x, save_path / "cells_before_ml_x.pkl.gz")
    save_compressed_pickle(groups_combined, save_path / "groups_combined.pkl.gz")
    save_compressed_pickle(spots_id_dic, save_path / "spots_id_dic.pkl.gz")
    save_compressed_pickle(spots_id_dic_prop, save_path / "spots_id_dic_prop.pkl.gz")
    save_compressed_pickle(pct_toml_dic, save_path / 'pct_toml_dic.pkl.gz')
    
    # Perform optimization
    spot_cell_dic = su.start_optimization(
        spots_X_dic,
        celltypes_dic,
        cells_X_plus_dic,
        nonzero_indices_toml,
        config.device,
        num_epochs=1000,
        learning_rate=0.1,
        print_each=10,
        epsilon=0.00001
    )
    
    # Save optimization results
    with open(save_path / 'spot_cell_dic.pkl', 'wb') as f:
        pickle.dump(spot_cell_dic, f)
    
    # Generate final data
    adata_sc_final = su.get_finaldata(
        adata,
        adatas_final,
        spot_cell_dic,
        weights_record,
        cells_before_ml,
        groups_combined,
        pct_toml_dic,
        nonzero_indices_dic,
        spots_X_dic,
        cells_before_ml_x=cells_before_ml_x,
        so=so
    )
    
    # Save final results
    adata_sc_final.write(str(config.output_dir / 'cells_smurf.h5ad'))

def main() -> None:
    """Main entry point."""
    try:
        config = parse_arguments()
        process_data(config)
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

if __name__ == "__main__":
    main()
