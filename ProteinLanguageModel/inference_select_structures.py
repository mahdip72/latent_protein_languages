"""Select one sampled structure per input using internal entropy and token consensus."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import yaml

from utils.inference.confidence_selection import (
    _parse_metric_list,
    _resolve_deepconf_selection_config,
    compute_deepconf_entropy_scores,
    _select_deepconf_candidate,
)


def select(input_path: str, output_path: str, config_path: str) -> int:
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError('Input and output must be different files.')
    with open(config_path, encoding='utf-8') as handle:
        cfg = _resolve_deepconf_selection_config(yaml.safe_load(handle))
    if not cfg['enabled']:
        raise ValueError('deepconf_selection.enabled must be true.')
    groups = defaultdict(list)
    with open(input_path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {'prediction_id', 'predicted_structures', 'predicted_structure_token_entropy_raw'}
        if not required.issubset(fields):
            raise ValueError(f'Prediction CSV needs columns {sorted(required)}.')
        for row in reader:
            base_id, separator, generation = row['prediction_id'].rpartition('__gen_')
            if not separator or not generation.isdigit():
                raise ValueError('prediction_id must contain the __gen_INDEX suffix from multi-sample inference.')
            entropy = _parse_metric_list(row['predicted_structure_token_entropy_raw'])
            scores = compute_deepconf_entropy_scores(entropy, cfg)
            scores['mean_structure_entropy_raw'] = sum(entropy)/len(entropy) if entropy else float('nan')
            codes = [int(code) for code in row['predicted_structures'].split()]
            if any(code < 0 or code >= 4096 for code in codes):
                raise ValueError('Predicted structure content codes must be in [0, 4095].')
            groups[base_id].append({**scores, '_structure_tokens': codes, '_row': row})
    selected = []
    for base_id, candidates in groups.items():
        chosen = _select_deepconf_candidate(candidates, cfg['score'], cfg)
        if chosen is None:
            raise ValueError(f'No finite confidence scores for {base_id}.')
        selected.append({**chosen['_row'], 'selection_score': chosen[cfg['score']]})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, 'selection_score'])
        writer.writeheader()
        writer.writerows(selected)
    return len(selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Multi-sample predicted_structures.csv')
    parser.add_argument('--output', required=True, help='Selected prediction CSV')
    parser.add_argument('--config_path', '-c', default='configs/inference/confidence_selection.yaml')
    args = parser.parse_args()
    print(f'Selected {select(args.input, args.output, args.config_path)} predictions.')
