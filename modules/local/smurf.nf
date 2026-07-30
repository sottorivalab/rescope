process  smurf {
    tag "Perform smurf soft segmentation for ${sample_id}"
    publishDir "${params.outdir}/${sample_id}", mode: 'copy'
    input:
    tuple val(sample_id), 
          path(bins_adata), 
          path(bins_parquet), 
          path(cells_json),
          path(he_zarr),
          path(he_tiff)
    
    output:
    tuple val(sample_id), 
          path('segmentation.npy.gz'), 
          path('smurf_intermediate'), 
          emit: for_pp
    path('cells_smurf.h5ad')
    
    conda params.env
    script:
    """
    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib/
    export PYTHONPATH=${projectDir}:\${PYTHONPATH:-}
    python ${projectDir}/src/smurf_process.py \
        --bins_parquet $bins_parquet \
        --bins_adata $bins_adata \
        --cellvit_json $cells_json \
        --zarr_path $he_zarr \
        --im_path $he_tiff \
    """
}