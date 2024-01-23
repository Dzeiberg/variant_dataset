"""
Utilities to read MutPred2 outputs;
currently assumes each job is in a separate directory
and that the job contains mutations for a single sequence.

Author : Daniel Zeiberg
Edit Date : January 2024
email : zeiberg.d@northeastern.edu
"""
from pathlib import Path
import os
import contextlib
from tqdm import tqdm
from fire import Fire
import scipy.io
import numpy as np
import joblib
import pandas as pd
import logging
from data_utils import Protein, Variant

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def read_features(job_dir):
    try:
        features = scipy.io.loadmat(Path(job_dir) / 'output.txt.feats_1.mat')['feats']
    except Exception as e:
        raise e
    return features

def read_substitutitons(job_dir):
    substitutions = np.array(list(map(lambda a: a.item().item(), 
        scipy.io.loadmat(Path(job_dir) / 'output.txt.substitutions.mat')['substitutions'].item())))
    return substitutions

def read_sequence(job_dir) -> str:
    sequence = scipy.io.loadmat(Path(job_dir) / 'output.txt.sequences.mat')['sequences'].item().item()
    return sequence

def get_scores(job_dir,substitutions):
    job_dir = Path(job_dir)
    # map of substitution string to score
    score_dict = pd.read_csv(job_dir / 'output.txt').set_index('Substitution')['MutPred2 score'].to_dict()
    scores = np.array(list(map(lambda s: score_dict[s], substitutions)))
    return scores

def process_job(job_dir):
    features = read_features(job_dir)
    substitutions = read_substitutitons(job_dir)
    sequence = read_sequence(job_dir)
    scores = get_scores(job_dir,substitutions)
    return features, substitutions, sequence,scores

def single_job(job_dir):
    try:
        features_, substitutions_, sequence_,scores_ = process_job(job_dir)
        return features_, substitutions_, sequence_,scores_
    except Exception as e:
        logging.warning(f'Error processing job {job_dir}\n{e}')
        # print(f'Error processing job {job_dir} {e}')
        return None, None, None,None
    
def file_check(job_path):
    if not os.path.isfile(job_path / 'output.txt'):
        return False
    if not os.path.isfile(job_path / 'output.txt.feats_1.mat'):
        return False
    if not os.path.isfile(job_path / 'output.txt.substitutions.mat'):
        return False
    if not os.path.isfile(job_path / 'output.txt.sequences.mat'):
        return False
    return True

def check_job_num(d,max_job_num):
    if max_job_num is None:
        return True
    job_num = int(d.split("_")[-1])
    if job_num > max_job_num:
        return False
    else:
        return True

def read_mutpred2_results(results_base_dirs, max_job_num=None,output_filepath=None,only_count_jobs=False,limit=None):
    jobs = []
    for results_base_dir in results_base_dirs:
        results_base_dir = Path(results_base_dir)
        jobs_ = [results_base_dir / d for d in os.listdir(results_base_dir) if file_check(results_base_dir / d) and check_job_num(d,max_job_num)]
        jobs.extend(jobs_)
    jobs.sort(key=lambda s: int(str(s).split("_")[-1]))
    if only_count_jobs:
        print(f'Found {len(jobs)} jobs')
        print("\n".join(list(map(str,jobs))))
        return

    if limit is not None:
        jobs = jobs[:limit]
    
    with tqdm_joblib(tqdm(desc="Processing Jobs", total=len(jobs))) as progress_bar:
        job_results = joblib.Parallel(n_jobs=16)(joblib.delayed(single_job)(job_dir) for job_dir in jobs)
    assert len(jobs) == len(job_results), f"{len(jobs)} job paths and {len(job_results)} results"
    proteins = {}
    for job_num,(job_path,(features_, substitutions_,sequence_,scores_)) in tqdm(enumerate(zip(jobs, job_results)),total=len(jobs)):
        if features_ is None:
            logging.warning(f"features is None Skipping job {job_path}")
            continue
        if  substitutions_ is None:
            logging.warning(f"substitutions is None Skipping job {job_path}")
            continue
        if sequence_ is None:
            logging.warning(f"sequence is None Skipping job {job_path}")
            continue
        if scores_ is None:
            logging.warning(f"scores is None Skipping job {job_path}")
            continue
        try:
            protein = proteins[Protein.get_md5_hash(sequence_)]
            # logging.warning(f"found existing protein with {len(protein.variants)} variants")
            new_protein = False
        except KeyError:
            protein = Protein(sequence_, sequence_)
            new_protein = True
        for feat_, sub_, score_ in zip(features_, substitutions_, scores_):
            aa_ref, aa_pos, aa_alt = sub_[0], int(sub_[1:-1]), sub_[-1]
            try:
                variant = Variant(protein,aa_ref, aa_pos, aa_alt,)
            except ValueError as e:
                logging.warning(f"Skipping variant {sub_} for protein {protein.id}\n{protein}\n: {e}")
                continue
            variant.add_feature_vec(feat_)
            variant.add_annotation("MutPred2_Score", score_)
            variant.add_annotation("local_job_path", str(job_path))
            protein.add_variant(variant)
        if new_protein:
            assert sequence_ not in set(proteins.keys())
            proteins[Protein.get_md5_hash(sequence_)] = protein
    # proteins = list(proteins.values())
    if output_filepath is not None:
        joblib.dump(list(proteins.values()),output_filepath)
    return proteins

if __name__ == '__main__':
    Fire(read_mutpred2_results)