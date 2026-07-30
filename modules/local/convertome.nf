process  convertOME {
    tag "Processing tiff>ome.zarr>ome.tiff for ${sample_id}"
    publishDir "${params.outdir}/${sample_id}", mode: 'copy'

    input:
    tuple val(sample_id), 
          path(composite)
    
    output:
    tuple val(sample_id), 
          path('composite.ome.zarr'), 
          path('composite.ome.tiff') 
    
    conda params.env
    script:
    """
    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib/
    bioformats2raw $composite composite.ome.zarr
    raw2ometiff composite.ome.zarr composite.ome.tiff
    """
}