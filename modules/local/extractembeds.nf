process  extractEmbeds {
    tag "Embeddings tensor extraction for ${sample_id}"
    publishDir "${params.outdir}/${sample_id}", mode: 'copy'

    input:
    tuple val(sample_id), path(cells_embed)
    
    output:
    tuple val(sample_id), path('cells_image_embeddings.pt'), path('cells_image_positions.pt'), path('cells_image_metadata.json')
    
    conda params.env
    script:
    """
    python3 ${projectDir}/src/extract_embeds.py --cellvit_res $cells_embed
    """
}