process  bin2cell {
    tag "Processing bin to cell mapping for ${sample_id}"
    publishDir "${params.outdir}/${sample_id}", mode: 'copy'

    input:
    tuple val(sample_id), 
          path(bins_adata), 
          path(scale_fac),
          path(bins_parquet), 
          path(cells_json)
    
    output:
    tuple val(sample_id), 
          path('cells_b2c.h5ad'), 
          path('cells_b2c.parquet')
    
    conda params.env
    script:
    """
    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib/
    python ${projectDir}/src/bin2cell.py \
        --bins_parquet $bins_parquet \
        --bins_adata $bins_adata \
        --cellvit_json $cells_json \
        --scale_fac $scale_fac \
        --expand_labels ${params.expand_labels}
    """
}