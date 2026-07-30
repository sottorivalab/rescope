process  CellVitPlusPlusDetection {
    tag "Running CV++ segmentation for ${sample_id}"
    publishDir "${params.outdir}/${sample_id}", mode: 'copy'

    input:
    tuple val(sample_id), path(image, stageAs: "image.tiff")
    
    output:
    tuple path('image/cell_detection.geojson'), 
        path('image/cell_detection.json'), 
        path('image/cells.geojson')
    tuple val(sample_id), 
        path('image/cells.json'),
        emit: cells_json
    tuple val(sample_id),
        path('image/cells.pt'),
        emit: cells_embed
    
    conda params.env
    script:
    """
    export CUDA_LAUNCH_BLOCKING=1
    export CUDA_VISIBLE_DEVICES=0
    cellvit-inference \
            --model SAM \
            --outdir ./ \
            --geojson \
            --graph \
            --nuclei_taxonomy ${params.model_name} \
            --ray_worker 1 \
            process_wsi \
            --wsi_path $image \
            --wsi_mpp 0.25 \
            --wsi_magnification 40 
    """
    // wsi_properties required by CV++
}