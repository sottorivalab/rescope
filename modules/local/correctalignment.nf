process correctAlignment {
    tag "Calculating alignment correction for ${sample_id}"
    publishDir "${params.outdir}/${sample_id}/alignment", mode: 'copy'

    input:
    tuple val(sample_id), 
          path(bins_adata, stageAs: "bins_2um.h5"), 
          path(bins_parquet, stageAs: "bins_2um.parquet"), 
          path(scale_fac),
          path(he_zarr)
    
    output:
    tuple val(sample_id), 
          path('transforms.npy'),
          emit: alignment_results
    tuple val(sample_id), 
          path('composite_corrected.tif'),
          emit: composite_results
    tuple val(sample_id),
          path('bins_corrected.parquet'),
          emit: corrected_bins
    path('bins_raster_raw.npy')
    path('hematoxylin_raw.npy')
    
    conda params.env
    
    script:
    """
    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib/
    python ${projectDir}/src/correct_alignment.py \
        --bins_adata bins_2um.h5 \
        --bins_parquet bins_2um.parquet \
        --scale_fac $scale_fac \
        --he_zarr $he_zarr \
        --microns_per_pixel ${params.img_mpp} \
        --image_level ${params.img_level} \
    """
}