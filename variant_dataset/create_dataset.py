from igvf_catalog.data_utils import Protein, Variant, Dataset, Labeler
from igvf_catalog.read_reference_databases import process_clinvar, process_gnomad, read_gnomad_and_ensembl
from igvf_catalog.read_mutpred2_results import read_mutpred2_results
from pathlib import Path
from tqdm import tqdm
import logging
from collections import defaultdict

def get_disease_gene_hashes(clinvar):
    disease_gene_hashes = set()
    for protein in clinvar.values():
        for variant in protein.variants.values():
            if variant.get_annotation("clinvar20231213_ClinicalSignificance") in set(("Pathogenic",
                                                                                        "Pathogenic/Likely pathogenic",
                                                                                        "Likely pathogenic")) and variant.get_annotation("clinvar20231213_star_rating") >= 1:
                disease_gene_hashes.add(protein.sequence_hash)
                break
    return disease_gene_hashes

def create_dataset():
    logging.info("Reading gnomAD")
    gnomad = process_gnomad(*read_gnomad_and_ensembl())
    logging.info("Reading ClinVar")
    clinvar = process_clinvar()
    disease_gene_hashes = get_disease_gene_hashes(clinvar)
    
    gnomad_disease_genes = {seq_hash : protein for seq_hash,protein in gnomad.items() if seq_hash in disease_gene_hashes}
    logging.info("Reading MutPred2 Featurized Data")
    proteins_with_mutpred_results = get_proteins_with_mutpred2_results()
    logging.info("Done reading data")
    labeler = Labeler(clinvar,gnomad_disease_genes)
    dataset = Dataset()
    for p in proteins_with_mutpred_results.values():
        for v in p.variants.values():
            v.add_annotation("clinvar_annotation", labeler.get_clinvar_status(v))
            v.add_annotation("in_gnomad", labeler.check_gnomad_status(v))
            v.add_annotation("my_label", labeler.label_variant(v))
            try:
                for annotation_name, annotation_value in gnomad[v.protein.sequence_hash].variants[v.substitution_str].annotations.items():
                    v.add_annotation(annotation_name, annotation_value)
            except KeyError:
                pass
            try:
                for annotation_name, annotation_value in clinvar[v.protein.sequence_hash].variants[v.substitution_str].annotations.items():
                    v.add_annotation(annotation_name, annotation_value)
            except KeyError:
                pass
            if v.get_annotation("clinvar_annotation") != "clinvar other/not present" or v.get_annotation("in_gnomad"):
                dataset.add_variant(v)
    return dataset

def get_proteins_with_mutpred2_results():
    rt=Path("/data/projects/igvf/catalog/mutpred2_predictions/phases/bilabel")
    proteins_with_mutpred_results = read_mutpred2_results(["/data/projects/igvf/catalog/mutpred2_predictions/phases/phase1/",
                                                            "/data/projects/igvf/catalog/mutpred2_predictions/phases/phase1.5/"]+ 
                                                            [rt / d for d in "clinvar_all_variants_jobs all_missing gnomad_hgmd_other_jobs missing_jobs jobs".split()])
    return proteins_with_mutpred_results

def count_variant_annotations(dataset):
    annotation_counts = defaultdict(int)
    for variant in dataset.variants:
        clinvar_status = variant.get_annotation("clinvar_annotation")
        gnomad_status = variant.get_annotation("in_gnomad")
        if clinvar_status in set(("P/LP", "B/LB")):
            annotation_counts[clinvar_status] += 1
        if gnomad_status:
            annotation_counts['gnomad'] += 1
    return annotation_counts