process  postprocess_smurf {
    tag "Restore full transcriptome ${sample_id}"
    publishDir "${params.outdir}/${sample_id}", mode: 'copy'
    input:
    tuple val(sample_id), 
          path(bins_adata), 
          path(bins_parquet), 
          path(segmentation),
          path(intermediate_results),
          path(he_tiff)
    
    output:
    tuple val(sample_id), 
          path('cells_smurf_full.h5ad')
    
    conda params.env
    script:
    """
    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib/
    export PYTHONPATH=${projectDir}:\${PYTHONPATH:-}
    python ${projectDir}/src/smurf_postprocess.py \
        --bins_parquet $bins_parquet \
        --bins_adata $bins_adata \
        --segmentation $segmentation \
        --intermediate_results $intermediate_results \
        --im_path $he_tiff \
    """
}