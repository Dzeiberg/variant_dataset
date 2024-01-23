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
from igvf_catalog.data_utils import Protein, Variant
from Bio.PDB.Polypeptide import protein_letters_3to1

server = "https://rest.ensembl.org"
ext = "/archive/id"
headers={ "Content-Type" : "application/json", "Accept" : "application/json"}


def read_gzip_fasta(path):
    records = {}
    with gzip.open(path,'rt') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            records[record.id] = str(record.seq)
    return records

def read_gnomad(path="/data/dbs/gnomad/v2_liftover/gnomad.exomes.r2.1.1.sites.missense.PASS.pkl"):
    gnomad = pd.read_pickle(path)
    return gnomad

def read_gnomad_and_ensembl(path="/data/dbs/gnomad/v2_liftover/gnomad.exomes.r2.1.1.sites.missense.PASS.pkl",
                ensemble_path="/data/dbs/ensembl/Homo_sapiens.GRCh38.pep.all.fa.gz",
                cached_ensembl_file="/home/dzeiberg/igvf_catalog/ensembl_records.pkl"):
    residues = set(protein_letters_3to1.values())
    gnomad = read_gnomad(path)
   
    ensembl_records = read_gzip_fasta(ensemble_path)
    # protein_ids_in_gnomad = set(gnomad.HGVSp.str.split(":").str[0].values)
    # missing_ids = set(protein_ids_in_gnomad) - set(ensembl_records.keys())
    # n_queries = 1 + (len(missing_ids) // 1000)
    # for subset in tqdm(np.array_split(list(missing_ids), n_queries),total=n_queries):
    #     r = requests.post(server+ext, headers=headers, data=repr({"id":list(map(lambda s: s.split(".")[0],subset))}).replace("\'","\""))
    #     for query, hit in zip(subset,r.json()):
    #         seq = hit['peptide']
    #         if seq is None or any([r not in residues for r in seq]):
    #             raise ValueError(f'could not match {query}\n{hit}')
    #         ensembl_records[query] = hit['peptide']
    #         logging.warning(f">{query}\n{hit['peptide']}\n")
    gnomad = gnomad[gnomad.HGVSp != "."]
    return gnomad, ensembl_records

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
                    mane_fasta_path="/data/dbs/mane/MANE.GRCh38.v1.1.refseq_protein.faa.gz"):
    clinvar_df = pd.read_pickle(clinvar_pickle_path)
    mane_records = read_gzip_fasta(mane_fasta_path)
    clinvar_proteins = dict()
    for refseq_prot_id, variants in clinvar_df.groupby("MANE_RefSeq_prot"):
        protein = Protein(refseq_prot_id, mane_records[refseq_prot_id])
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
        clinvar_proteins[Protein.sequence_hash] = protein
    return clinvar_proteins
