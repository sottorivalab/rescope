# adapted code from https://github.com/Teichlab/bin2cell
from typing import Tuple, Dict, Optional
import argparse
import anndata as ad
import pandas as pd
import geopandas as gpd
import json
import numpy as np
from pathlib import Path
import scanpy as sc
import scipy
import shapely
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
from tqdm import tqdm

# Constants
MIN_CELLS = 3
MIN_COUNTS = 1
AMBIGUOUS_BIN_THRESHOLD = 0.8
MIN_BINS_PER_CELL = 4
MAX_BINS_PER_CELL = 150
DEFAULT_K_NEIGHBORS = 4
DEFAULT_VOLUME_RATIO = 4
DEFAULT_MAX_BIN_DISTANCE = 2
DISTANCE_MASK = 1000

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Postprocess')
    parser.add_argument('-ba', '--bins_adata',
                       type=str, required=True,
                       help='bins_adata path')
    parser.add_argument('-bp', '--bins_parquet',
                       type=str, required=True,
                       help='bins_parquet path')
    parser.add_argument('-sf', '--scale_fac',
                       type=str, required=True,
                       help='bins scale factors')
    parser.add_argument('-cj', '--cellvit_json',
                       type=str, required=True,
                       help='path with cellvit results')
    parser.add_argument('-oa', '--output_adata',
                       type=str, default='cells_b2c.h5ad',
                       help='output AnnData path')
    parser.add_argument('-op', '--output_parquet',
                       type=str, default='cells_b2c.parquet',
                       help='output parquet path')
    parser.add_argument('-tx', '--transforms',
                       type=str, required=False,
                       help='complex transformations to apply')
    parser.add_argument('-dx', '--delta_x',
                       type=str, default='0',
                       help='x offset')
    parser.add_argument('-dy', '--delta_y',
                       type=str, default='0',
                       help='y offset')
    parser.add_argument('-a', '--alignment',
                       type=str, required=False,
                       help='alignment correction JSON file')
    parser.add_argument('-el', '--expand_labels',
                       type=bool, required=False, default=True,
                       help='whether to expand labels')
    args = parser.parse_args()
    if args.alignment:
        with open(args.alignment, 'r') as f:
            alignment = json.load(f)
            args.delta_x = alignment.get('dx', 0)
            args.delta_y = alignment.get('dy', 0)

    return args

def destripe_counts(adata, counts_key="n_counts",
                    adjusted_counts_key="n_counts_adjusted"):
    '''
    Scale each row (bin) of ``adata.X`` to have ``adjusted_counts_key`` 
    rather than ``counts_key`` total counts.

    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw counts, needs to have ``counts_key`` 
        and ``adjusted_counts_key`` in ``.obs``.
    counts_key : ``str``, optional (default: ``"n_counts"``)
        Name of ``.obs`` column with raw counts per bin.
    adjusted_counts_key : ``str``, optional (default: ``"n_counts_adjusted"``)
        Name of ``.obs`` column storing the desired destriped counts per bin.
    '''
    # scanpy's utility function to make sure the anndata is not a view
    # if it is a view then weird stuff happens when you try to write to its .X
    sc._utils.view_to_actual(adata)
    # adjust the count matrix to have n_counts_adjusted sum per bin (row)
    # premultiplying by a diagonal matrix multiplies each row by a value: https://solitaryroad.com/c108.html
    bin_scaling = scipy.sparse.diags(adata.obs[adjusted_counts_key]/adata.obs[counts_key])
    adata.X = bin_scaling.dot(adata.X)


def destripe(adata,
             quantile=0.99,
             counts_key="n_counts",
             factor_key="destripe_factor",
             adjusted_counts_key="n_counts_adjusted",
             adjust_counts=True):
    ''' 
    Correct the raw counts of the input object for known variable width of 
    VisiumHD 2um bins. Scales the total UMIs per bin on a per-row and 
    per-column basis, dividing by the specified ``quantile``. The resulting 
    value is stored in ``.obs[factor_key]``, and is multiplied by the 
    corresponding total UMI ``quantile`` to get ``.obs[adjusted_counts_key]``.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw counts, needs to have ``counts_key`` in 
        ``.obs``.
    quantile : ``float``, optional (default: 0.99)
        Which row/column quantile to use for the computation.
    counts_key : ``str``, optional (default: ``"n_counts"``)
        Name of ``.obs`` column with raw counts per bin.
    factor_key : ``str``, optional (default: ``"destripe_factor"``)
        Name of ``.obs`` column to hold computed factor prior to reversing to 
        count space.
    adjusted_counts_key : ``str``, optional (default: ``"n_counts_adjusted"``)
        Name of ``.obs`` column for storing the destriped counts per bin.
    adjust_counts : ``bool``, optional (default: ``True``)
        Whether to use the computed adjusted count total to adjust the counts in 
        ``adata.X``.
    '''
    # apply destriping via sequential quantile scaling
    # get specified quantile per row
    quant = adata.obs.groupby("array_row")[counts_key].quantile(quantile)
    # divide each row by its quantile (order of obs[counts_key] and obs[array_row] match)
    adata.obs[factor_key] = adata.obs[counts_key] / adata.obs["array_row"].map(quant)
    # repeat on columns
    quant = adata.obs.groupby("array_col")[factor_key].quantile(quantile)
    adata.obs[factor_key] /= adata.obs["array_col"].map(quant)
    # propose adjusted counts as the global quantile multipled by the destripe factor
    adata.obs[adjusted_counts_key] = adata.obs[factor_key] * np.quantile(adata.obs[counts_key], quantile)
    # correct the count space unless told not to
    if adjust_counts:
        destripe_counts(adata, counts_key=counts_key, adjusted_counts_key=adjusted_counts_key)


def process_cell_json(cells_json: Path) -> Tuple[Dict[str, str], gpd.GeoDataFrame]:
    """Process cell detection JSON file and create GeoDataFrame.
    
    Args:
        cells_json: Path to the cells.json file
        
    Returns:
        Tuple containing:
        - type_map dictionary
        - GeoDataFrame with cell data
    """
    with open(cells_json, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            raise

    type_map_inverse = {v: k for k, v in data['type_map'].items()} if 'type_map' in data.keys() else None
    
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
            "type": cell["type"] if 'type' in cell.keys() else np.nan,
            "type_label": data['type_map'][str(cell["type"])] if 'type_map' in data.keys() else np.nan,
            "type_prob": cell["type_prob"] if 'type_prob' in cell.keys() else np.nan,
            "bbox": cell["bbox"] if 'bbox' in cell.keys() else np.nan,
            "centroid": cell["centroid"] if 'centroid' in cell.keys() else [geometry.centroid.x, geometry.centroid.y],
            "patch_coordinates": cell["patch_coordinates"] if 'patch_coordinates' in cell.keys() else np.nan,
            "cell_status": cell["cell_status"] if 'cell_status' in cell.keys() else np.nan,
            "offset_global": cell["offset_global"] if 'offset_global' in cell.keys() else np.nan,
            "edge_position": cell["edge_position"] if 'edge_position' in cell.keys() else np.nan,
        }
        cell_records.append(cell_record)
    
    return type_map_inverse, gpd.GeoDataFrame(cell_records).set_geometry("geometry")


def process_bins_pd(bins_path: str, scale_fac: str, dx: float = 0, dy: float = 0, tx=None) -> Tuple[gpd.GeoDataFrame, float]:
    """Process bins data.
    
    Args:
        bins_path: Path to bins parquet file
        scale_fac: Path to scale factors JSON
        dx: X-axis offset
        dy: Y-axis offset
        
    Returns:
        Tuple containing:
        - Processed bins GeoDataFrame
        - bin area value
    """
    positions = pd.read_parquet(bins_path)
    positions.loc[:,'pxl_col_in_fullres'] = positions['pxl_col_in_fullres'].values + dx
    positions.loc[:,'pxl_row_in_fullres'] = positions['pxl_row_in_fullres'].values + dy
    
    # Calculate rotation angle
    TL = positions.nsmallest(1, ['array_col']).iloc[0]
    TR = positions.nlargest(1, ['array_col']).nsmallest(1, ['array_row']).iloc[0]
    delta_x = TL['pxl_col_in_fullres'] - TR['pxl_col_in_fullres']
    delta_y = TL['pxl_row_in_fullres'] - TR['pxl_row_in_fullres']
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    
    positions = positions.query("in_tissue==1")
    
    with open(scale_fac, 'r') as f:
        sc_fac = json.load(f)
    spot_size = sc_fac['spot_diameter_fullres']
    bin_area = spot_size**2
    
    # Create geometry
    positions.loc[:, 'geometry'] = (gpd.points_from_xy(positions['pxl_col_in_fullres'], 
                                              positions['pxl_row_in_fullres'])
                           .buffer(spot_size / 2, cap_style=3))
    bins = gpd.GeoDataFrame(positions, geometry='geometry')
    bins.loc[:, 'geometry'] = bins['geometry'].apply(
        lambda geom: shapely.affinity.rotate(geom, angle_deg, origin='centroid')
    )
    if tx is not None:
        tx = np.load(tx)
        bins['geometry'] = bins.geometry.apply(lambda geom: affine_transform(geom, tx))
    bins.loc[:, 'geometry_bins'] = bins.geometry.copy()
    
    return bins.set_geometry("geometry"), bin_area
    

def resolve_cell_bin_conflicts(
    joined: gpd.GeoDataFrame, 
    bin_area: float,
    ambiguous_bin_thr: float = AMBIGUOUS_BIN_THRESHOLD,
    min_bins_per_cell: int = MIN_BINS_PER_CELL,
    max_bins_per_cell: int = MAX_BINS_PER_CELL
) -> pd.Series:
    """Resolve conflicts in cell-to-bin assignment.
    
    Args:
        joined: GeoDataFrame with joined cell and bin data
        bin_area: Area of a single bin
        ambiguous_bin_thr: Threshold for ambiguous bin assignment
        min_bins_per_cell: Minimum bins per cell
        max_bins_per_cell: Maximum bins per cell
        
    Returns:
        Series mapping location_ids to cell_ids
    """
    conflicts = joined.barcode.value_counts()
    conflicts = conflicts.loc[conflicts > 1]
    
    joined_to_solve = joined.loc[joined.barcode.isin(conflicts.index)].copy()
    joined = joined.loc[~joined.barcode.isin(conflicts.index)]
    
    joined_to_solve.loc[:, 'intersection'] = joined_to_solve['geometry'].intersection(
        joined_to_solve['geometry_bins']
    )
    joined_to_solve = joined_to_solve[
        joined_to_solve['intersection'].area > ambiguous_bin_thr * bin_area
    ]
    
    joined = pd.concat([joined.barcode, joined_to_solve.barcode])
    joined = joined.loc[
        (joined.index.value_counts() >= min_bins_per_cell) & 
        (joined.index.value_counts() <= max_bins_per_cell)
    ]
    
    return pd.Series(index=joined.values, data=joined.index)


def expand_labels(adata, 
                labels_key="labels", 
                expanded_labels_key="labels_expanded", 
                algorithm="max_bin_distance", 
                max_bin_distance=DEFAULT_MAX_BIN_DISTANCE,
                volume_ratio=DEFAULT_VOLUME_RATIO, 
                k=DEFAULT_K_NEIGHBORS, 
                subset_pca=True):
    '''
    Expand segmentation results to bins a maximum distance away in 
    the array coordinates. In the event of multiple equidistant bins with 
    different labels, ties are broken by choosing the closest bin in a PCA 
    representation of gene expression. The resulting labels will be integers, 
    with 0 being unassigned to an object.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw or destriped counts.
    labels_key : ``str``, optional (default: ``"labels"``)
        ``.obs`` key holding the labels to be expanded. Integers, with 0 being 
        unassigned to an object.
    expanded_labels_key : ``str``, optional (default: ``"labels_expanded"``)
        ``.obs`` key to store the expanded labels under.
    algorithm : ``str``, optional (default: ``"max_bin_distance"``)
        Toggle between ``max_bin_distance`` or ``volume_ratio`` based label 
        expansion.
    max_bin_distance : ``int`` or ``None``, optional (default: 2)
        Maximum number of bins to expand the nuclear labels by.
    volume_ratio : ``float``, optional (default: 4)
        A per-label expansion distance will be proposed as 
        ``ceil((volume_ratio**(1/3)-1) * sqrt(n_bins/pi))``, where 
        ``n_bins`` is the number of bins for the corresponding pre-expansion 
        label. Default based on cell line 
        `data <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8893647/>`_
    k : ``int``, optional (default: 4)
        Number of assigned spatial coordinate bins to find as potential nearest 
        neighbours for each unassigned bin.
    subset_pca : ``bool``, optional (default: ``True``)
        If ``True``, will obtain the PCA representation of just the bins 
        involved in the tie breaks rather than the full bin space. Results in 
        a slightly different embedding at a lower resource footprint.
    '''
    #this is where the labels will go
    adata.obs[labels_key] = adata.obs[labels_key].astype(int)
    adata.obs[expanded_labels_key] = adata.obs[labels_key].values.copy()
    #get out our array grid, and preexisting labels
    coords = adata.obs[["array_row","array_col"]].values
    labels = adata.obs[labels_key].values
    #we'll be splitting the space in two - the bins with labels, and those without
    object_mask = (labels != 0)
    #get their indices in cell space
    full_reference_inds = np.arange(adata.shape[0])[object_mask]
    full_query_inds = np.arange(adata.shape[0])[~object_mask]
    #for each unassigned bin, we'll find its k nearest neighbours in the assigned space
    #build a reference using the assigned bins' coordinates
    ckd = scipy.spatial.cKDTree(coords[object_mask, :])
    #query it using the unassigned bins' coordinates
    dists, hits = ckd.query(x=coords[~object_mask,:], k=k, workers=-1)
    #convert the identified indices back to the full cell space
    hits = full_reference_inds[hits]
    #get the label calls for each of the hits
    calls = labels[hits]
    #get the area (bin count) of each object
    label_values, label_counts = np.unique(labels, return_counts=True)
    #this is how the algorithm was toggled early on
    #switched to an argument to avoid potential future spaghetti
    if max_bin_distance is None:
        raise ValueError("Use ``algorithm`` to toggle between algorithms")
    if algorithm == "volume_ratio":
        #compute the object's sphere's radius as sqrt(nbin/pi)
        #scale to radius of cell by multiplying by volume_ratio^(1/3)
        #and subtract away the original radius to account for presence of nucleus
        #do a ceiling to compensate for possible reduction of area in slice
        label_distances = np.ceil((volume_ratio**(1/3)-1) * np.sqrt(label_counts/np.pi))
        #get an array where you can index on object and get the distance
        #needs +1 as the max value of label_values is actually present in the data
        label_distance_array = np.zeros((np.max(label_values)+1,))
        label_distance_array[label_values] = label_distances
    elif algorithm == "max_bin_distance":
        #just use the provided value
        label_distance_array = np.ones((np.max(label_values)+1,)) * max_bin_distance
    else:
        raise ValueError("``algorithm`` must be ``'max_bin_distance'`` or ``'volume_ratio'``")
    #construct a matching dimensionality array of max distance allowed per call
    max_call_distance = label_distance_array[calls]
    #mask bins too far away from call with arbitrary high value
    dist_mask = 1000
    dists[dists > max_call_distance] = dist_mask
    #evaluate the minima in each row. start by getting said minima
    min_per_bin = np.min(dists, axis=1)[:,None]
    #now get positions in each row that have the minimum (and aren't the mask)
    is_hit = (dists == min_per_bin) & (min_per_bin < dist_mask)
    #case one - we have a solitary hit of the minimum
    clear_mask = (np.sum(is_hit, axis=1) == 1)
    #get out the indices of the bins
    clear_query_inds = full_query_inds[clear_mask]
    #np.argmin(axis=1) finds the column of the minimum per row
    #subsequently retrieve the matching hit from calls
    clear_query_labels = calls[clear_mask, np.argmin(dists[clear_mask, :], axis=1)]
    #insert calls into object
    adata.obs.loc[adata.obs_names[clear_query_inds], expanded_labels_key] = clear_query_labels
    #case two - 2+ assigned bins are equidistant
    ambiguous_mask = (np.sum(is_hit, axis=1) > 1)
    if np.sum(ambiguous_mask) > 0:
        #get their indices in the original cell space
        ambiguous_query_inds = full_query_inds[ambiguous_mask]
        if subset_pca:
            #in preparation of PCA, get a master list of all the bins to PCA
            #we've got two sets - the query bins, and their k hits
            #the hits needs to be .flatten()ed after masking to become 1d again
            #np.unique sorts in an ascending fashion, which is convenient
            smol = np.unique(np.concatenate([hits[ambiguous_mask,:].flatten(), ambiguous_query_inds]))
            #prepare a PCA as a representation of the GEX space for solving ties
            #can just run straight on an array to get a PCA matrix back. convenient!
            #keep the object's X raw for subsequent cell creation
            pca_smol = sc.pp.pca(np.log1p(adata.X[smol, :]))
            #mock up a "full-scale" PCA matrix to not have to worry about different indices
            pca = np.zeros((adata.shape[0], pca_smol.shape[1]))
            pca[smol, :] = pca_smol
        else:
            #just run a full space PCA
            pca = sc.pp.pca(np.log1p(adata.X))
        #compute the distances between the expression profiles of the undecided bin and the neighbours
        #np.linalg.norm is the fastest way to get euclidean, subtract two point sets beforehand
        #pca[hits[ambiguous_mask, :]] is bins by k by num_pcs
        #pca[ambiguous_query_inds, :] is bins by num_pcs
        #add the [:, None, :] and it's bins by 1 by num_pcs, and subtracts as you'd hope
        eucl_input = pca[hits[ambiguous_mask, :]] - pca[ambiguous_query_inds, :][:, None, :]
        #can just do this along axis=2 and get all the distances at once
        eucl_dists = np.linalg.norm(eucl_input, axis=2)
        #mask ineligible bins with arbitrary high value
        eucl_mask = 1000
        eucl_dists[~is_hit[ambiguous_mask, :]] = eucl_mask
        #define calls based on euclidean minimum
        #same argmin/mask logic as with clear before
        ambiguous_query_labels = calls[ambiguous_mask, np.argmin(eucl_dists, axis=1)]
        #insert calls into object
        adata.obs.loc[adata.obs_names[ambiguous_query_inds], expanded_labels_key] = ambiguous_query_labels

        
def create_cell_adata(
    adata: ad.AnnData, 
    cell_mapping: pd.Series, 
    cells: gpd.GeoDataFrame,
    apply_labels_expansion: bool = True
) -> ad.AnnData:
    """Create cell AnnData object from bin data.
    
    Args:
        adata: Original bin AnnData object
        cell_mapping: Series mapping location_ids to cell_ids
        cells: GeoDataFrame with cell data
        apply_labels_expansion: Whether to apply label expansion
        
    Returns:
        New AnnData object with cell data
    """
    adata.obs = adata.obs.assign(cell_id=adata.obs.index.map(cell_mapping.to_dict()))
    adata.obs.loc[:, 'cell_id'] = adata.obs['cell_id'].fillna(0).astype(int)
    
    if apply_labels_expansion:
        expand_labels(adata, 'cell_id', "cell_id_expanded")
        adata = adata[adata.obs['cell_id_expanded'] > 0]
        cell_to_bin = pd.get_dummies(adata.obs['cell_id_expanded'], sparse=True)
    else:
        adata = adata[adata.obs['cell_id'] > 0]
        cell_to_bin = pd.get_dummies(adata.obs['cell_id'], sparse=True)
    
    cell_names = [str(i) for i in cell_to_bin.columns]
    cell_to_bin = cell_to_bin.sparse.to_coo().tocsr().T
    X = cell_to_bin.dot(adata.X).tocsr()
    
    cell_adata = ad.AnnData(X, var=adata.var)
    cell_adata.obs_names = cell_names
    
    bin_count = np.asarray(cell_to_bin.sum(axis=1)).flatten()
    row_means = scipy.sparse.diags(1/bin_count)
    
    cell_adata.obs['bin_count'] = bin_count
    cell_adata.obs["array_row"] = row_means.dot(cell_to_bin).dot(adata.obs["array_row"].values)
    cell_adata.obs["array_col"] = row_means.dot(cell_to_bin).dot(adata.obs["array_col"].values)

    ci = cell_adata.obs.index.astype(float).astype(int)
    cell_adata.obsm['spatial'] = np.array([
        [x[0], x[1]] for x in cells.loc[ci, 'centroid'].values
    ])
    
    cell_adata.obs.loc[cell_adata.obs.index, ['type_label', 'type_prob']] = (
        cells.loc[ci, ['type_label', 'type_prob']].values
    )
    
    return cell_adata

def main(to_destripe=False):
    """Main function to process and create cell data."""
    args = parse_arguments()
    
    # Load and process cell detection results
    _, gdf = process_cell_json(Path(args.cellvit_json))
    
    # Process bins data
    bins, bin_area = process_bins_pd(
        args.bins_parquet, 
        args.scale_fac, 
        dx=float(args.delta_x), 
        dy=float(args.delta_y)
    )
    
    # Perform spatial join and resolve conflicts
    joined = gpd.sjoin(gdf, bins)
    cell_mapping = resolve_cell_bin_conflicts(joined, bin_area)
    
    # Create cell AnnData object
    adata = sc.read_10x_h5(args.bins_adata)
    adata.var_names_make_unique()
    adata.obs.loc[bins.barcode.values,
        ['array_row', 'array_col', 
         'pxl_row_in_fullres', 'pxl_col_in_fullres']] = (
        bins.reset_index()
            .set_index('barcode')[['array_row', 'array_col',
                                 'pxl_row_in_fullres', 'pxl_col_in_fullres']]
            .values
    )
    
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    sc.pp.filter_cells(adata, min_counts=MIN_COUNTS)
    if to_destripe:
        destripe(adata)
    
    cell_adata = create_cell_adata(adata, cell_mapping, gdf, apply_labels_expansion=args.expand_labels)
    cell_adata.write_h5ad(args.output_adata)
    joined.to_parquet(args.output_parquet)

if __name__ == "__main__":
    main()
