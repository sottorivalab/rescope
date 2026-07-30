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
    segmentation: Path
    intermediate_results: Path
    im_path: Path
    output_dir: Path

def parse_arguments() -> SmurfConfig:
    """Parse command line arguments into a SmurfConfig object."""
    parser = argparse.ArgumentParser(description='restore full exps')
    
    # Required arguments
    parser.add_argument('-ba', '--bins_adata', type=str, required=True,
                       help='Path to bins AnnData file')
    parser.add_argument('-bp', '--bins_parquet', type=str, required=True,
                       help='Path to bins parquet file')
    parser.add_argument('-s', '--segmentation', type=str, required=True,
                       help='path to labeled image')
    parser.add_argument('-ir', '--intermediate_results', type=str,
                        required=True, help='path to smurf outs')
    parser.add_argument('-ip', '--im_path', type=str, required=True,
                         help='Path to image')

    # Optional arguments
    parser.add_argument('-od', '--output_dir', type=str, default='./',
                       help='Output path')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    config = SmurfConfig(
        bins_adata=Path(args.bins_adata),
        bins_parquet=Path(args.bins_parquet),
        segmentation=Path(args.segmentation),
        intermediate_results=Path(args.intermediate_results),
        im_path=Path(args.im_path),
        output_dir=Path(args.output_dir),
    )
    return config
    
def load_compressed_pickle(filename):
    with gzip.GzipFile(filename, 'rb') as f:
        return pickle.load(f)
def save_compressed_pickle(data: any, filename: PathLike) -> None:
    """Save data as a compressed pickle file."""
    with gzip.GzipFile(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    config = parse_arguments()
    so = su.prepare_dataframe_image(
        config.bins_parquet,
        config.im_path,
        'HE'
    )
    with gzip.GzipFile(config.segmentation, 'r') as f:
        raster = np.load(f)
    so.segmentation_final = raster.copy()
    so.generate_cell_spots_information()
    adata = sc.read_10x_h5(str(config.bins_adata))
    adata = adata[so.df[so.df.in_tissue == 1]['barcode']].copy()
    sc.pp.filter_genes(adata, min_counts=1000)
    su.nuclei_rna(adata, so)

    cells_before_ml = load_compressed_pickle(config.intermediate_results / 'cells_before_ml.pkl.gz')
    groups_combined = load_compressed_pickle(config.intermediate_results / 'groups_combined.pkl.gz')
    pct_toml_dic = load_compressed_pickle(config.intermediate_results / 'pct_toml_dic.pkl.gz')
    nonzero_indices_dic = load_compressed_pickle(config.intermediate_results / 'nonzero_indices_dic.pkl.gz')
    nonzero_indices_toml = load_compressed_pickle(config.intermediate_results/ 'nonzero_indices_toml.pkl.gz')
    with open(config.intermediate_results / 'cells_final.pkl', 'rb') as file:
        cells_final = pickle.load(file)
    with open(config.intermediate_results / 'weights_record.pkl', 'rb') as file:
        weights_record = pickle.load(file)
    with open(config.intermediate_results / 'spot_cell_dic.pkl', 'rb') as f:
        spot_cell_dic = pickle.load(f)
    adatas_final = sc.read_h5ad(config.intermediate_results / 'adatas.h5ad')
    
    adata1 = sc.read_10x_h5(config.bins_adata) 
    adata1 = adata1[so.df[so.df.in_tissue == 1]['barcode']] 
    weight_to_celltype = su.calculate_weight_to_celltype(adatas_final, adata1, cells_final, so) 
    save_compressed_pickle(weight_to_celltype, "weight_to_celltype.pkl.gz")
    
    adata_sc_final2 = su.get_finaldata(adata1, 
                                   adatas_final, 
                                   spot_cell_dic, 
                                   weight_to_celltype, 
                                   cells_before_ml, 
                                   groups_combined, 
                                   pct_toml_dic, 
                                   nonzero_indices_dic, 
                                   spots_X_dic = None, 
                                   nonzero_indices_toml = nonzero_indices_toml, 
                                   cells_before_ml_x = None, 
                                   so=so)
    adata_sc_final2.write(str(config.output_dir / 'cells_smurf_full.h5ad'))
    
if __name__ == "__main__":
    main()
