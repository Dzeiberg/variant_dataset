import hashlib
import itertools
from tqdm import tqdm
import logging
import numpy as np
from Bio.PDB.Polypeptide import protein_letters_3to1

class Protein(object):
    def __init__(self, id, sequence):
        self.id = id
        self.sequence = sequence
        self.length = len(sequence)
        self.variants = []
        self.existing_substitution_set = set()

    def get_missing_variants(self):
        missing_variants = []
        for position, ref_aa in enumerate(self.sequence,start=1):
            for alt_aa in set(protein_letters_3to1.values()) - set([ref_aa]):
                substitution_str = ref_aa+str(position)+alt_aa
                if substitution_str not in self.existing_substitution_set:
                    missing_variants.append(substitution_str)
        return missing_variants

    def add_external_id(self, external_id_type, external_id):
        if not hasattr(self, "external_ids"):
            self.external_ids = {}
        self.external_ids[external_id_type] = external_id

    def add_variant(self, variant):
        substitution_str = variant.aa_ref+str(variant.aa_pos)+variant.aa_alt
        if substitution_str in self.existing_substitution_set:
            return
        self.variants.append(variant)
        self.existing_substitution_set.add(substitution_str)

    def __repr__(self):
        return f">{self.id}\n{self.sequence}\n"
    
    @classmethod
    def get_md5_hash(cls,s):
        md5_hash = hashlib.md5()
        md5_hash.update(s.encode())
        return md5_hash.hexdigest()

    @property
    def sequence_hash(self):
        return self.get_md5_hash(self.sequence)

class Variant(object):
    def __init__(self, protein : Protein, aa_ref : str, aa_pos : int, aa_alt : str):
        self.protein = protein
        self.aa_ref = aa_ref
        self.aa_pos = aa_pos
        self.aa_alt = aa_alt
        self.validate_variant()
        self._feature_vec = None

    def add_feature_vec(self, feature_vec):
        self._feature_vec = feature_vec.reshape((1,-1))
    
    def __repr__(self):
        return f"PROTEIN{self.protein.id}:{self.aa_ref}{self.aa_pos}{self.aa_alt}"

    def get_feature_vec(self):
        if self._feature_vec is None:
            raise ValueError("Feature vector not set")
        return self._feature_vec

    def validate_variant(self):
        if self.aa_pos > self.protein.length:
            raise ValueError(f"Variant position {self.aa_pos} is greater than protein length {self.protein.length}")
        if self.aa_ref != self.protein.sequence[self.aa_pos-1]:
            raise ValueError(f"Variant reference amino acid {self.aa_ref} does not match protein sequence {self.protein.sequence[self.aa_pos-1]}")

    def add_annotation(self, annotation_name, annotation_value):
        if not hasattr(self, "annotations"):
            self.annotations = {}
        self.annotations[annotation_name] = annotation_value

    def get_annotation(self, annotation_name):
        if not hasattr(self, "annotations"):
            raise ValueError("No annotations set")
        try:
            return self.annotations[annotation_name]
        except KeyError:
            return np.nan

def process_results(results, existing_proteins=None):
    if existing_proteins is None:
        existing_proteins = dict()
    variants = sorted(zip(results['sequence_ids'], results['substitutions'], results['features'], results['scores']),
                        key=lambda x: x[0])
    for sequence_id, variants in tqdm(itertools.groupby(variants, key=lambda x: x[0]),total=len(results['sequence_map'])):
        try:
            protein = existing_proteins[results['sequence_map'][sequence_id]]
            new_protein = False
        except KeyError:
            protein = Protein(sequence_id, results['sequence_map'][sequence_id])
            new_protein = True
        for _, substitution, feature_vec, score in variants:
            aa_ref, aa_pos, aa_alt = substitution[0], int(substitution[1:-1]), substitution[-1]
            try:
                variant = Variant(protein,aa_ref, aa_pos, aa_alt,)
            except ValueError as e:
                logging.warning(f"Skipping variant {substitution} for protein {protein.id}\n{protein}\n: {e}")
                continue
            variant.add_feature_vec(feature_vec)
            variant.add_annotation("MutPred2_Score", score)
            protein.add_variant(variant)
        if new_protein:
            existing_proteins[protein.sequence_hash] = protein
    return existing_proteins