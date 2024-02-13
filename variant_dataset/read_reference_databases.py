import logging
import pandas as pd
from pandas import DataFrame
import gzip
from Bio import SeqIO
import requests, sys
import numpy as np
from tqdm import tqdm
import os
import joblib
from variant_dataset.data_utils import Protein, Variant
from Bio.PDB.Polypeptide import protein_letters_3to1
import urllib

logging.basicConfig(filename='/tmp/log.txt', encoding='utf-8', level=logging.DEBUG)



def read_gzip_fasta(path):
    records = {}
    with gzip.open(path,'rt') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            records[record.id] = str(record.seq)
    return records

def process_gnomad(df, ensembl_records):
    gnomad_proteins = dict()
    for Ensembl_protein_id, variants in tqdm(df.groupby("Ensembl_protein_id"),total=len(df.Ensembl_protein_id.unique())):
        try:
            wt_seq = ensembl_records[Ensembl_protein_id]
        except KeyError:
            logging.warning(f"{Ensembl_protein_id} not present in ensembl records")
            continue
        protein = Protein(Ensembl_protein_id, wt_seq)
        for _, row in variants.iterrows():
            # row["HGCSp"] looks like ":p.Ala123Val"
            try:
                substitution = row["HGVSp"].split(":")[1][2:]
                try:
                    aa_ref = protein_letters_3to1[substitution[:3].upper()]
                    aa_alt = protein_letters_3to1[substitution[-3:].upper()]
                except KeyError as e:
                    logging.warning(f"Skipping variant {row['HGVSp']} for protein {protein.id}\n{protein}\n: {e}")
                    continue
                aa_pos = int(substitution[3:-3])
                variant = Variant(protein,aa_ref, aa_pos, aa_alt)
            except ValueError as e:
                logging.warning(f"Skipping variant {row['HGVSp']} for protein {protein.id}\n{protein}\n: {e}")
                continue
            for annotation_name,annotation_value in row.items():
                variant.add_annotation("gnomad_v211_" + annotation_name, annotation_value)
            protein.add_variant(variant)
        gnomad_proteins[protein.sequence_hash] = protein
    return gnomad_proteins


def read_gnomad_vcf_files(*args,**kwargs):
    gnomad_exomes = kwargs.get("gnomad_exomes",'/mnt/i/bio/gnomad/v2_liftover/gnomad.exomes.r2.1.1.sites.missense.vcf.gz')
    gnomad_genomes = kwargs.get("gnomad_genomes",'/mnt/i/bio/gnomad/v2_liftover/gnomad.genomes.r2.1.1.sites.missense.vcf.gz')
    header_file = kwargs.get("gnomad_header_file",'/mnt/i/bio/gnomad/v2_liftover/header.txt')

    with open(header_file) as f:
        columns = f.read().strip().split(",")
    
    exomes = pd.read_csv(gnomad_exomes,delimiter='\t',header=None,compression='gzip')
    exomes.columns = columns
    genomes = pd.read_csv(gnomad_genomes,delimiter='\t',header=None,compression='gzip')
    genomes.columns = columns

    gnomad = pd.concat((exomes,genomes))#.drop_duplicates(subset=['HGVSp'])
    gnomad = gnomad.assign(Ensembl_protein_id = gnomad.HGVSp.str.split(":").str[0])
    gnomad.AF = pd.to_numeric(gnomad.AF,errors='coerce')
    return gnomad

def process_clinvar(clinvar_variant_summary,mane_summary,mane_records, disease_genes):
    mane_summary = mane_summary.reset_index().set_index("GeneID")
    clinvar_snvs = clinvar_variant_summary[clinvar_variant_summary.Type == "single nucleotide variant"]
    clinvar_snvs = clinvar_snvs.assign(GeneID = clinvar_snvs.GeneID.astype(str))
    clinvar_snvs = clinvar_snvs[clinvar_snvs.GeneID.isin(disease_genes.GeneID)]
    clinvar_snvs = clinvar_snvs[clinvar_snvs.Name.str.contains("p.")]
    clinvar_snvs = clinvar_snvs.assign(HGVSp=":p."+clinvar_snvs.Name.str.split("p.").str[1].str.slice(0,-1))
    clinvar_proteins = dict()
    for GeneID, variants in tqdm(clinvar_snvs.groupby("GeneID"),total=len(clinvar_snvs.GeneID.unique())):
        try:
            RefSeq_protein_id = mane_summary.loc[GeneID, "RefSeq_prot"]
        except KeyError as e:
            logging.warning(f"Skipping protein {GeneID}: {e}")
            continue
        try:
            wt_seq = mane_records[RefSeq_protein_id]
        except KeyError as e:
            logging.warning(f"Skipping protein {GeneID}: {e}")
            continue
        protein = Protein(RefSeq_protein_id, wt_seq)
        for _, row in variants.iterrows():
            try:
                substitution = row["HGVSp"].split(":")[1][2:]
                try:
                    aa_ref = protein_letters_3to1[substitution[:3].upper()]
                    aa_alt = protein_letters_3to1[substitution[-3:].upper()]
                except KeyError as e:
                    logging.warning(f"Skipping variant {row['HGVSp']} for protein {protein.id}\n{protein}\n: {e}")
                    continue
                aa_pos = int(substitution[3:-3])
                variant = Variant(protein,aa_ref, aa_pos, aa_alt)
            except ValueError as e:
                logging.warning(f"Skipping variant {row['HGVSp']} for protein {protein.id}\n{protein}\n: {e}")
                continue
            for annotation_name,annotation_value in row.items():
                variant.add_annotation("clinvar" + annotation_name, annotation_value)
            protein.add_variant(variant)
        clinvar_proteins[protein.sequence_hash] = protein
    return clinvar_proteins

def clinvar_pathogenicity_status(row,pathogenic_or_benign):
    if pathogenic_or_benign == "pathogenic":
        return row.ClinicalSignificance in ["Pathogenic","Likely pathogenic", "Pathogenic/Likely pathogenic"] and \
            row.ReviewStatus not in {'no clasification provided', 'no assertion criteria provided','no classification for the single variant'}

    elif pathogenic_or_benign == 'benign':
        return row.ClinicalSignificance in ["Benign", "Likely benign", "Benign/Likely benign"] and \
            row.ReviewStatus not in {'no clasification provided', 'no assertion criteria provided','no classification for the single variant'}
    else:
        raise ValueError()

def get_disease_genes(clinvar_variant_summary, mane_summary):
    disease_variants = clinvar_variant_summary[clinvar_variant_summary.is_pathogenic]
    disease_variants = disease_variants.assign(GeneID = disease_variants.GeneID.astype(str))
    disease_genes = mane_summary[mane_summary.GeneID.isin(disease_variants.GeneID)]
    return disease_genes
    
def remove_clinvar_pathogenic_and_benign_from_gnomad(clinvar_proteins, gnomad_proteins):
    for seq_hash,p in gnomad_proteins.items():
        if seq_hash in clinvar_proteins:
            for substitution_str in clinvar_proteins[seq_hash].variants:
                if substitution_str in p.variants and \
                    (clinvar_proteins[seq_hash].variants[substitution_str].is_pathogenic or \
                        clinvar_proteins[seq_hash].variants[substitution_str].is_benign):
                    p.variants.pop(substitution_str)

def read_steps(*args, **kwargs):
    """
    Required kwargs:
    ensembl_fasta : str : e.g. path to 'Homo_sapiens.GRCh38.pep.all.fa.gz'
    gnomad_exomes : str : e.g. path to gnomad.exomes.r2.1.1.sites.missense.vcf.gz : expect gzip tsv filepath
    gnomad_genomes : str : e.g. path to gnomad.genomes.r2.1.1.sites.missense.vcf.gz : expect gzip tsv filepath
    gnomad_header_file : str : path to column names in the gnomad files
    """
    clinvar_variant_summary = pd.read_csv("https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",delimiter="\t",compression='gzip')
    clinvar_variant_summary = clinvar_variant_summary.assign(is_pathogenic=clinvar_variant_summary.apply(lambda row: clinvar_pathogenicity_status(row,'pathogenic'),axis=1),
                                                                is_benign=clinvar_variant_summary.apply(lambda row: clinvar_pathogenicity_status(row,'benign'),axis=1))

    mane_summary = pd.read_csv("https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.2/MANE.GRCh38.v1.2.summary.txt.gz",compression='gzip',delimiter='\t')
    mane_summary = mane_summary[mane_summary.MANE_status == "MANE Select"]
    mane_summary = mane_summary.assign(GeneID=mane_summary.loc[:,"#NCBI_GeneID"].str.split(":").str[1])

    urllib.request.urlretrieve("https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.2/MANE.GRCh38.v1.2.refseq_protein.faa.gz",'/tmp/MANE.GRCh38.v1.2.refseq_protein.faa.gz')
    mane_records = read_gzip_fasta('/tmp/MANE.GRCh38.v1.2.refseq_protein.faa.gz')
    # ~1 min , returns 4251 genes
    # columns '#NCBI_GeneID, Ensembl_Gene, HGNC_ID, symbol, name, RefSeq_nuc, RefSeq_prot, Ensembl_nuc, Ensembl_prot, MANE_status, GRCh38_chr, chr_start, chr_end, chr_strand, GeneID'
    disease_genes = get_disease_genes(clinvar_variant_summary,mane_summary)
    gnomad_genomes_and_exomes = read_gnomad_vcf_files(**kwargs)
    # 912304 unique variants (from HGVSp)
    gnomad_sample = gnomad_genomes_and_exomes[(gnomad_genomes_and_exomes.FILTER == 'PASS') & \
                                              (gnomad_genomes_and_exomes.AF < 0.01) & \
                                                (gnomad_genomes_and_exomes.Ensembl_protein_id.isin(disease_genes.Ensembl_prot))]
    ensembl = read_gzip_fasta(kwargs.get("ensembl_fasta",'/mnt/i/bio/ensembl/Homo_sapiens.GRCh38.pep.all.fa.gz'))
    # 849254 variants, 2747 genes
    gnomad_proteins = process_gnomad(gnomad_sample, ensembl)
    # Keep only pathogenic and benign
    clinvar_variant_summary = clinvar_variant_summary[clinvar_variant_summary.is_pathogenic | clinvar_variant_summary.is_benign]
    clinvar_proteins = process_clinvar(clinvar_variant_summary,mane_summary,mane_records,disease_genes)
    # remove_clinvar_pathogenic_and_benign_from_gnomad(clinvar_proteins, gnomad_proteins)
    return clinvar_proteins, gnomad_proteins