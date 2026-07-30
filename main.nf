#!/usr/bin/env nextflow

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES / SUBWORKFLOWS / FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
include { CellVitPlusPlusDetection } from './modules/local/cellvitpp.nf'
include { extractEmbeds } from './modules/local/extractembeds.nf'
include { correctAlignment } from './modules/local/correctalignment.nf'
include { bin2cell } from './modules/local/bin2cell.nf'
include { smurf } from './modules/local/smurf.nf'
include { postprocess_smurf } from './modules/local/postprocesssmurf.nf'
include { convertOME } from './modules/local/convertome.nf'

workflow {
    // Create channel from CSV file with explicit file naming
    Channel
        .fromPath(params.input_csv)
        .splitCsv(header:true)
        .map { row -> tuple(
            row.sample_id,
            file(row.bins_adata, checkIfExists: true),
            file(row.bins_parquet, checkIfExists: true),
            file(row.scale_fac, checkIfExists: true),
            file(row.he_zarr, checkIfExists: true),
            file(row.he_tiff, checkIfExists: true),
            row.custom_cells_json ? file(row.custom_cells_json, checkIfExists: true) : null,
            row.correct_alignment ? row.correct_alignment.toBoolean() : true
        )}
        .set { samples_ch }

    // Run CellVit if enabled
    if (params.segment_with_cvpp) { 
        CellVitPlusPlusDetection(
            samples_ch.map { sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment -> 
                tuple(sample_id, he_tiff) 
            }
        )
    }

    // Extract embeddings
    if (params.extract_embeddings && params.segment_with_cvpp) {
        extractEmbeds(
            CellVitPlusPlusDetection.out.cells_embed
        )
    }

    // Calculate alignment correction for samples that need it
    correctAlignment(
        samples_ch
            .filter { sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment -> 
                correct_alignment 
            }
            .map { sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment ->
                tuple(sample_id, bins_adata, bins_parquet, scale_fac, he_zarr)
            }
    )

    // Create a channel for corrected bins (either from alignment or original)
    corrected_bins_ch = samples_ch
        .map { sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment ->
            if (correct_alignment) {
                tuple(sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment, true)
            } else {
                tuple(sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment, false)
            }
        }
        .join(correctAlignment.out.corrected_bins, by: [0, 0], remainder: true)
        .map { sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment, needs_correction, corrected_bins ->
            if (needs_correction) {
                tuple(sample_id, corrected_bins)
            } else {
                tuple(sample_id, bins_parquet)
            }
        }
    // Create a channel for cell JSON files (either from CV++ or custom)
    cells_json_ch = params.segment_with_cvpp ? 
        CellVitPlusPlusDetection.out.cells_json :
        samples_ch.map { sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment ->
            tuple(sample_id, custom_json)
        }.filter { sample_id, json -> json != null }

    // Process with either bin2cell or smurf based on the selected method
    if (params.processing_method == 'bin2cell' || params.processing_method == 'both') {
        bin2cell_input_ch = samples_ch.map { sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment ->
            tuple(sample_id, bins_adata, scale_fac)
        }
        .join(corrected_bins_ch)
        .join(cells_json_ch)
        bin2cell(bin2cell_input_ch)
    } 
    if (params.processing_method == 'smurf' || params.processing_method == 'both') {
        // Create intermediate channels for smurf process
        smurf_bins_ch = samples_ch.map { sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment ->
            tuple(sample_id, bins_adata)
        }
        
        smurf_he_ch = samples_ch.map { sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment ->
            tuple(sample_id, he_zarr, he_tiff)
        }
        
        // Join channels in sequence
        smurf_input_ch = smurf_bins_ch
            .join(corrected_bins_ch)
            .join(cells_json_ch)
            .join(smurf_he_ch)
            .map { sample_id, bins_adata, bins_parquet, cells_json, he_zarr, he_tiff ->
                tuple(sample_id, bins_adata, bins_parquet, cells_json, he_zarr, he_tiff)
            }
        smurf(smurf_input_ch)
        pp_input_ch  = smurf_bins_ch
                        .join(corrected_bins_ch)
                        .join(smurf_he_ch)
                        .join(smurf.out.for_pp)
                        .map { sample_id, bins_adata, bins_parquet,
                        he_zarr, he_tiff, segmentation, smurf_intermediate ->
                        tuple(sample_id, bins_adata, bins_parquet, 
                        segmentation, smurf_intermediate, he_tiff)
                        }
        postprocess_smurf(pp_input_ch)
    }
    // Convert OME for samples that had alignment correction
    convertOME(
        samples_ch
            .filter { sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment -> 
                correct_alignment 
            }
            .map { sample_id, bins_adata, bins_parquet, scale_fac, he_zarr, he_tiff, custom_json, correct_alignment ->
                tuple(sample_id)
            }
            .join(correctAlignment.out.composite_results)
    )
}
