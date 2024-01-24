from igvf_catalog.data_utils import Protein, Variant, Dataset, Labeler
from igvf.read_reference_databases import process_clinvar, process_gnomad, read_gnomad_and_ensembl
from igvf.read_mutpred2_results import read_mutpred2_results
from pathlib import Path
from tqdm import tqdm

def create_dataset():
    clinvar = process_clinvar()
    gnomad = process_gnomad()
    proteins_with_mutpred_results = get_proteins_with_mutpred2_results()
    labeler = Labeler(clinvar,gnomad)
    dataset = Dataset()
    for p in proteins_with_mutpred_results.values():
        for v in p.variants.values():
            v.add_annotation("my_label", labeler.label_variant(v))
            if v.get_annotation("my_label") != "not present":
                dataset.add_variant(v)
    return dataset

def get_proteins_with_mutpred2_results():
    rt=Path("/data/projects/igvf/catalog/mutpred2_predictions/phases/bilabel")
    proteins_with_mutpred_results = read_mutpred2_results(["/data/projects/igvf/catalog/mutpred2_predictions/phases/phase1/",
                                                            "/data/projects/igvf/catalog/mutpred2_predictions/phases/phase1.5/"]+ 
                                                            [rt / d for d in "clinvar_all_variants_jobs all_missing gnomad_hgmd_other_jobs missing_jobs jobs".split()])
    return proteins_with_mutpred_results
