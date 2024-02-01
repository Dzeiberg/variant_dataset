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
logging.basicConfig(filename='/home/dzeiberg/variant_dataset/log.txt', encoding='utf-8', level=logging.DEBUG)
server = "https://rest.ensembl.org"
ext = "/archive/id"
headers={ "Content-Type" : "application/json", "Accept" : "application/json"}


def read_gzip_fasta(path):
    records = {}
    with gzip.open(path,'rt') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            records[record.id] = str(record.seq)
    return records

def read_gnomad(path="/data/dzeiberg/gene_calibration/gnomad.genomes_and_exomes.r2.1.1.sites.pkl"):
    gnomad = pd.read_pickle(path)
    return gnomad

def read_gnomad_and_ensembl(path="/data/dzeiberg/gene_calibration/gnomad.genomes_and_exomes.r2.1.1.sites.pkl",
                ensemble_path="/data/dbs/ensembl/Homo_sapiens.GRCh38.pep.all.fa.gz",
                cached_ensembl_file="/home/dzeiberg/variant_dataset/ensembl_records.pkl"):
    gnomad = read_gnomad(path)
   
    ensembl_records = read_gzip_fasta(ensemble_path)
    gnomad = gnomad[gnomad.HGVSp != "."]
    return gnomad, ensembl_records

def load_preprocessed_gnomad(gnomad_path="/data/dzeiberg/gene_calibration/vikas_calibration_gnomad_dataset.pkl",
                                ensemble_path="/data/dbs/ensembl/Homo_sapiens.GRCh38.pep.all.fa.gz"):
    gnomad = pd.read_pickle(gnomad_path)
    ensembl = read_gzip_fasta(ensemble_path)
    grouped = gnomad.groupby("Ensembl_proteinid_latest")
    proteins = {}
    for ensembl_transcript_id,variants in tqdm(grouped,total=len(grouped)):
        try:
            wt_seq = ensembl[ensembl_transcript_id]
        except KeyError:
            continue
        protein = Protein(ensembl_transcript_id, wt_seq)
        for _,row in variants.iterrows():
            aa_ref = row.aavar[0]
            aa_pos = int(row.aavar[1:-1])
            aa_alt = row.aavar[-1]
            try:
                variant = Variant(protein,aa_ref, aa_pos, aa_alt)
            except ValueError as e:
                print(e)
                continue
            for k,v in row.items():
                variant.add_annotation(k,v)
            protein.add_variant(variant)
        proteins[protein.sequence_hash] = protein
    return proteins


def process_gnomad(gnomad_df : DataFrame, ensembl_records : dict):
    gnomad_proteins = dict()
    gnomad_df = gnomad_df.assign(ensembl_transcript_id=gnomad_df.HGVSp.str.split(":").str[0])
    grouped = gnomad_df.groupby("ensembl_transcript_id")
    for ensembl_transcript_id,variants in tqdm(grouped,total=len(grouped)):
        try:
            wt_seq = ensembl_records[ensembl_transcript_id]
        except KeyError:
            continue
        protein = Protein(ensembl_transcript_id, wt_seq)
        for _, row in variants.iterrows():
            try:
                substitution = row["HGVSp"].split(":")[1][2:]
                try:
                    aa_ref = protein_letters_3to1[substitution[:3].upper()]
                    aa_alt = protein_letters_3to1[substitution[-3:].upper()]
                except KeyError as e:
                    # logging.warning(f"Skipping variant {row['HGVSp']} for protein {protein.id}\n{protein}\n: {e}")
                    continue
                aa_pos = int(substitution[3:-3])
                variant = Variant(protein,aa_ref, aa_pos, aa_alt)
            except ValueError as e:
                # logging.warning(f"Skipping variant {row['HGVSp']} for protein {protein.id}\n{protein}\n: {e}")
                continue
            for annotation_name,annotation_value in row.items():
                variant.add_annotation("gnomad_v211_" + annotation_name, annotation_value)
            protein.add_variant(variant)
        gnomad_proteins[protein.sequence_hash] = protein
    return gnomad_proteins

def process_clinvar(clinvar_pickle_path="/data/dbs/clinvar/clinvar_12_13_2023/variant_summary_annotated.pkl",
                    mane_fasta_path="/data/dbs/mane/MANE.GRCh38.v1.1.refseq_protein.faa.gz",
                    mane_summary_path="/data/dbs/mane/MANE.GRCh38.v1.1.summary.txt"):
    clinvar_df = pd.read_pickle(clinvar_pickle_path)
    mane_records = read_gzip_fasta(mane_fasta_path)
    mane_summary = pd.read_csv(mane_summary_path,sep="\t")
    mane_summary = mane_summary[mane_summary.MANE_status == "MANE Select"].set_index("symbol")
    clinvar_proteins = dict()
    grouped = clinvar_df.groupby("GeneSymbol")
    for gene_symbol, variants in tqdm(grouped, total=len(grouped)):
        try:
            canonical_RefSeq_prot = mane_summary.loc[gene_symbol.upper()].RefSeq_prot
            protein = Protein(gene_symbol, mane_records[canonical_RefSeq_prot])
        except KeyError as e:
            logging.warning(f"Skipping protein {gene_symbol}: {e}")
            continue
        for _, row in variants.iterrows():
            try:
                variant = Variant(protein,protein_letters_3to1[row["wt_aa"].upper()],
                                int(row['position']),
                                protein_letters_3to1[row["mt_aa"].upper()])
            except ValueError as e:
                logging.warning(f"Skipping variant {row['#AlleleID']} for protein {protein.id}\n{protein}\n: {e}")
                continue
            for annotation_name,annotation_value in row.items():
                variant.add_annotation("clinvar20231213_" + annotation_name, annotation_value)
            protein.add_variant(variant)
        clinvar_proteins[protein.sequence_hash] = protein
    return clinvar_proteins
