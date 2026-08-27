import numpy as np
import torch
import csv
from pathlib import Path
import json

from aac_metrics import evaluate
from aac_metrics.functional import fense, clap_sim, spice, cider_d
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from typing import List, Dict, Any, Union
import subprocess

def write_json(data: Union[List[Dict[str, Any]], Dict[str, Any]],
               path: Path) \
        -> None:
    """ Write a dict or a list of dicts into a JSON file

    :param data: Data to write
    :type data: list[dict[str, any]] | dict[str, any]
    :param path: Path to the output file
    :type path: Path
    """
    with path.open("w") as f:
        json.dump(data, f)

def evaluate_aac(candidates, multi_references, device, audio_paths=None):
    """
    Evaluate the AAC model on all metrics.

    Args:
        candidates (list): List of candidate texts.
        multi_references (list): List of reference texts.
        device (torch.device): Device to use for neural metrics.
        audio_paths (list): List of paths to audio files (required for clap_sim_audio).

    Returns:
        dict: {metric: {"score": float, "scores": [float, ...]}}
    """
    assert isinstance(candidates, list), "Candidates should be a list."
    assert isinstance(multi_references, list), "Multi-references should be a list."

    n = len(candidates)

    scores = defaultdict(dict)
    all_metrics = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor",
                   "cider_d", "spice", "spider", "fense", "clap_sim_text", "sbert_sim", "clap_sim_audio"]
    for metric in all_metrics:
        scores[metric]["score"] = 0.0
        scores[metric]["scores"] = []

    candidates = preprocess_mono_sents(candidates)
    multi_references = preprocess_mult_sents(multi_references)

    # Base metrics (no SPICE)
    base_metrics = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor"]
    other_scores_corpus, other_scores_sents = evaluate(candidates, multi_references, metrics=base_metrics)
    for metric in base_metrics:
        scores[metric]["scores"] = other_scores_sents[metric].tolist()
        scores[metric]["score"] = other_scores_corpus[metric].item()

    # SPICE with per-sample fallback on failure
    tqdm.write("Evaluating with SPICE...")
    try:
        spice_corpus, spice_sents = spice(candidates, multi_references, java_max_memory="16G")
        scores['spice']["scores"] = spice_sents['spice'].tolist()
    except Exception as e:
        tqdm.write(f"SPICE failed ({e}), retrying per sample...")
        per_sample_spice = []
        for i in range(n):
            print(f"Evaluating SPICE for sample {i+1}/{n}...")
            try:
                _, s_sent = spice([candidates[i]], [multi_references[i]], java_max_memory="16G")
                per_sample_spice.append(s_sent['spice'][0].item())
            except Exception:
                per_sample_spice.append(float('nan'))
                with open("spice_errors.log", "a") as log_f:
                    log_f.write(f"SPICE failed for sample {i}: Candidate='{candidates[i]}', References='{multi_references[i]}'\n")
        scores['spice']["scores"] = per_sample_spice

    # CIDEr-D
    tqdm.write("Evaluating with CIDEr-D...")
    cider_d_corpus, cider_d_sents = cider_d(candidates, multi_references)
    scores['cider_d']["scores"] = cider_d_sents['cider_d'].tolist()

    # SPIDEr
    tqdm.write("Evaluating with SPIDEr...")
    for i in range(n):
        s = scores['spice']["scores"][i]
        c = scores['cider_d']["scores"][i]
        if np.isnan(s) or np.isnan(c):
            scores['spider']["scores"].append(float('nan'))
        else:
            scores['spider']["scores"].append(0.5 * s + 0.5 * c)

    # FENSE
    tqdm.write("Evaluating with FENSE...")
    fense_score_corpus, fense_score_sents = fense(candidates, multi_references, device=device)
    scores['fense']["scores"] = fense_score_sents['fense'].tolist()
    scores['sbert_sim']["scores"] = fense_score_sents['sbert_sim'].tolist()

    # CLAP-sim text
    tqdm.write("Evaluating with CLAP-sim text...")
    clap_t_corpus, clap_t_sents = clap_sim(candidates, multi_references, device=device)
    scores['clap_sim_text']["scores"] = clap_t_sents['clap_sim'].tolist()

    # CLAP-sim audio
    if audio_paths is not None:
        tqdm.write("Evaluating with CLAP-sim audio...")
        clap_a_corpus, clap_a_sents = clap_sim(candidates, multi_references, clap_method="audio", device=device, audio_paths=audio_paths)
        scores['clap_sim_audio']["scores"] = clap_a_sents['clap_sim'].tolist()

    # NaN-safe corpus score aggregation
    for metric in scores.keys():
        arr = np.array(scores[metric]["scores"], dtype=float)
        scores[metric]["score"] = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0

    return scores

def evaluate_aac_batch(candidates, multi_references, batch_size, device, audio_paths=None, ignore_all=False):
    """
    Evaluate candidates against references on all 13 metrics, in batches.

    Args:
        candidates (list): List of candidate texts.
        multi_references (list): List of reference texts.
        device (torch.device) : Device to use for evaluation ('cpu' or 'cuda').
        audio_paths (list): List of paths to the audio files.
    
    Returns:
        dict: Dictionary containing the evaluation results.
    """
    # Ensure that candidates and multi_references are lists
    assert isinstance(candidates, list), "Candidates should be a list."
    assert isinstance(multi_references, list), "Multi-references should be a list."

    # Make a data loader for batch processing
    dataset = EvaluateDataset(candidates, multi_references, audio_paths if audio_paths is not None else [None]*len(candidates))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: (
        [item['candidate'] for item in x],
        [item['multi_reference'] for item in x],
        [item['audio_path'] for item in x] if audio_paths is not None else None
    ), num_workers=1)

    scores = defaultdict(dict)
    all_metrics = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor", "cider_d", "spice", "spider", "fense", "clap_sim_text", "sbert_sim", "clap_sim_audio"]
    for metric in all_metrics:
        scores[metric]["score"] = 0.0
        scores[metric]["scores"] = []
    
    # Evaluate traditional metrics
    metrics = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor"]

    for batch in tqdm(data_loader, desc="Evaluating batches"):
        batch_candidates, batch_multi_references, batch_audio_paths = batch

        batch_candidates = preprocess_mono_sents(batch_candidates)
        batch_multi_references = preprocess_mult_sents(batch_multi_references)

        try:
            tqdm.write("Evaluating with SPICE...")
            spice_score_corpus, spice_score_sents = spice(batch_candidates, batch_multi_references, java_max_memory="16G")
            scores['spice']["scores"].extend(spice_score_sents['spice'].tolist())
        except Exception as e:
            n = len(batch_candidates)

            if ignore_all:
                tqdm.write(f"SPICE batch failed ({e}), ignore_all=True — skipping entire batch.")
                for metric in all_metrics:
                    scores[metric]["scores"].extend([float('nan')] * n)
                continue

            tqdm.write(f"SPICE batch failed ({e}), retrying per sample...")

            # Retry SPICE per sample to isolate the problematic ones
            per_sample_spice = []
            for i in range(n):
                print(f"Evaluating SPICE for sample {i+1}/{n} in batch...")
                try:
                    _, s_sent = spice([batch_candidates[i]], [batch_multi_references[i]], java_max_memory="16G")
                    per_sample_spice.append(s_sent['spice'][0].item())
                except Exception:
                    per_sample_spice.append(float('nan'))
                    with open("spice_errors.log", "a") as log_f:
                        log_f.write(f"SPICE failed for sample {i} in batch: Candidate='{batch_candidates[i]}', References='{batch_multi_references[i]}'\n")
            valid_idx = [i for i, s in enumerate(per_sample_spice) if not np.isnan(s)]

            if not valid_idx:
                tqdm.write("All samples in batch failed SPICE. Skipping entire batch.")
                for metric in all_metrics:
                    scores[metric]["scores"].extend([float('nan')] * n)
                continue

            tqdm.write(f"{len(valid_idx)}/{n} samples recovered.")

            # Compute all other metrics only for the valid subset
            v_cands = [batch_candidates[i] for i in valid_idx]
            v_refs  = [batch_multi_references[i] for i in valid_idx]
            v_paths = [batch_audio_paths[i] for i in valid_idx] if audio_paths is not None else None

            # Build per-position result arrays initialised to NaN
            result = {m: [float('nan')] * n for m in all_metrics}
            for i, s in enumerate(per_sample_spice):
                result['spice'][i] = s

            oth_corp, oth_sent = evaluate(v_cands, v_refs, metrics=metrics)
            cid_corp, cid_sent = cider_d(v_cands, v_refs)
            fen_corp, fen_sent = fense(v_cands, v_refs, device=device)
            clap_corp, clap_sent = clap_sim(v_cands, v_refs, device=device)

            for j, i in enumerate(valid_idx):
                for m in metrics:
                    result[m][i] = oth_sent[m][j].item()
                result['cider_d'][i]      = cid_sent['cider_d'][j].item()
                result['spider'][i]       = 0.5 * per_sample_spice[i] + 0.5 * cid_sent['cider_d'][j].item()
                result['fense'][i]        = fen_sent['fense'][j].item()
                result['sbert_sim'][i]    = fen_sent['sbert_sim'][j].item()
                result['clap_sim_text'][i] = clap_sent['clap_sim'][j].item()

            if audio_paths is not None:
                clap_a_corp, clap_a_sent = clap_sim(v_cands, v_refs, clap_method="audio", device=device, audio_paths=v_paths)
                for j, i in enumerate(valid_idx):
                    result['clap_sim_audio'][i] = clap_a_sent['clap_sim'][j].item()

            for m in all_metrics:
                scores[m]["scores"].extend(result[m])
            continue

        other_scores_corpus, other_scores_sents = evaluate(batch_candidates, batch_multi_references, metrics=metrics)
        for metric in metrics:
            sent_scores = other_scores_sents[metric].tolist()
            scores[metric]["scores"].extend(sent_scores)
        
        # for metric in ["cider_d", "spice", "spider"]:
        #     sent_scores = other_scores_sents[metric].tolist()
        #     scores[metric]["scores"].extend(sent_scores)

        tqdm.write("Evaluating with CIDEr-d...")
        cider_d_score_corpus, cider_d_score_sents = cider_d(batch_candidates, batch_multi_references)
        sent_scores = cider_d_score_sents['cider_d'].tolist()
        scores['cider_d']["scores"].extend(sent_scores)

        tqdm.write("Evaluating with SPIDEr...")
        for i in range(len(batch_candidates)):
            spice_score = spice_score_sents['spice'][i].item()
            cider_d_score = cider_d_score_sents['cider_d'][i].item()
            spider_score = 0.5 * spice_score + 0.5 * cider_d_score
            scores['spider']["scores"].append(spider_score)

        tqdm.write("Evaluating with FENSE...")
        fense_score_corpus, fense_score_sents = fense(batch_candidates, batch_multi_references, device=device)
        sent_scores = fense_score_sents['fense'].tolist()
        scores['fense']["scores"].extend(sent_scores)

        sent_scores = fense_score_sents['sbert_sim'].tolist()
        scores['sbert_sim']["scores"].extend(sent_scores)

        tqdm.write("Evaluating with CLAP-sim text...")
        clap_sim_score_t_corpus, clap_sim_score_t_sents = clap_sim(batch_candidates, batch_multi_references, device=device)
        sent_scores = clap_sim_score_t_sents['clap_sim'].tolist()
        scores['clap_sim_text']["scores"].extend(sent_scores)

        if audio_paths is not None:
            tqdm.write("Evaluating with CLAP-sim audio")
            if "clap_sim_audio" not in scores:
                scores["clap_sim_audio"]["score"] = 0.0
                scores["clap_sim_audio"]["scores"] = []
            clap_sim_score_a_corpus, clap_sim_score_a_sents = clap_sim(batch_candidates, batch_multi_references, clap_method="audio", device=device, audio_paths=batch_audio_paths)
            sent_scores = clap_sim_score_a_sents['clap_sim'].tolist()
            scores['clap_sim_audio']["scores"].extend(sent_scores)
    
    # Average the corpus scores, ignoring NaN entries (failed SPICE samples)
    for metric in scores.keys():
        arr = np.array(scores[metric]["scores"], dtype=float)  # None -> NaN
        scores[metric]["score"] = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0
    return scores

class EvaluateDataset(Dataset):
    def __init__(self, candidates, multi_references, audio_paths):
        self.candidates = candidates
        self.multi_references = multi_references
        self.audio_paths = audio_paths

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, idx):
        return {
            "candidate": self.candidates[idx],
            "multi_reference": self.multi_references[idx],
            "audio_path": self.audio_paths[idx]
        }