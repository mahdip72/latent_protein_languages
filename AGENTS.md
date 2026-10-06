# Repository instructions for coding assistants

These instructions apply throughout this repository, regardless of model or assistant. Follow explicit task instructions and any more specific instructions in the folder you edit.

## Project and scope

This is the source release for *Learning Latent Protein Languages for Autoregressive Generation*. Read [README.md](README.md) for the paper and [EXPERIMENTS.md](EXPERIMENTS.md) for run lineage and reproduction limits.

- `ProteinLatentLanguage/` contains the PLL tokenizer, its two training stages, and sequence encoding/decoding.
- `ProteinLanguageModel/` contains amino acid PLM, PLLM, and SLLM training, length-conditioned continuation, E1-conditioned sequence-to-structure prediction, and inference selection.
- `provenance.json` records archived source revisions, configuration origins, source hashes, and release adaptations.
- SLL/GCP-VQVAE training and implementation belong to the separate [vq_encoder_decoder repository](https://github.com/mahdip72/vq_encoder_decoder).

Keep changes within the requested scope. This release excludes chemical models, ProteinBench pipelines, external baseline implementations, plotting scripts, datasets, weights, and raw run outputs. Do not introduce those components as part of an unrelated fix.

## Environment and entry points

Use Python 3.10 or later. Training targets Linux with CUDA-enabled PyTorch. Each project has its own `requirements.txt` and README. Use separate environments and run its commands from inside the corresponding project folder. Both projects use local `data`, `models`, and `utils` imports, so mixing their import paths can load the wrong module.

Typical training entry points, from the respective project folders:

```bash
accelerate launch --num_processes 1 train.py --config_path configs/stage1/config.yaml
accelerate launch --num_processes 1 train.py --config_path configs/scaling/aa/d512_l6/config.yaml
```

These are training commands, not smoke tests. GPU count and gradient accumulation affect the global batch. Consult the project guides before distributed training. E1 and GCP-VQVAE are additional dependencies for the relevant inference paths.

Do not start training, download large pretrained models, or launch distributed jobs for routine code or documentation checks. Use an existing suitable environment when available. Explain resource requirements before substantial execution requested by a task.

## Preserve experiment semantics

- Preserve saved training parameters and token IDs. Put new experiments in new configurations instead of silently rewriting the paper configurations.
- Load the saved `tokenizer_vocab.yaml`. The original amino acid language models use 20 canonical residues. The PLL decoder has a separate 21-class head including X.
- PLL has one content token per residue and a 4,096-state codebook. Its Stage 2 decoder is a reinitialized transformer with a linear residue head. The latent encoder and codebook are frozen. Do not replace this path with an MLP or train only the head while claiming paper compatibility.
- Keep stage-specific `learnable.yaml` files. Stage 1 uses stochastic code sampling and Stage 2 disables it.
- Structure token IDs require the codebook that produced the targets. Matching vocabulary sizes does not establish compatibility with another decoder.
- Preserve E1 sequence context in both generation and token scoring for sequence-to-structure inference.
- Keep confidence/consensus selection independent of reference structures and refolding quality scores.

The small matched tokenizer comparison and the larger E1-conditioned experiments are distinct. Do not substitute April 28 unconditional runs for the missing April 29/30 matched comparison snapshots. Unconfirmed checkpoint paths remain placeholders until supported by evidence.

## Editing and provenance

Inspect the current diff before editing and preserve unrelated work. Prefer small changes that match the existing style. Avoid broad formatting or architectural rewrites during a focused fix.

When changing a source file recorded in `provenance.json`, retain its original revision and source hash. Update its release hash and document the adaptation. Do not describe recovered compatible code as an exact per-run snapshot when the run did not record that evidence.

Keep documentation and configurations consistent with actual code. Do not infer measured results from configuration budgets, substitute validation metrics for test results, or change paper claims without supporting evidence. Keep uncertain details explicit in `EXPERIMENTS.md`.

Do not add credentials, private machine paths, confidential review correspondence, IDE state, model weights, or generated outputs to a public change. Keep dataset and output paths configurable. Preserve existing licenses and external model/data attribution.

## Validation

There is currently no repository-wide automated test suite. Use checks appropriate to the change and add focused regression tests when changing meaningful behavior. From the repository root, these lightweight checks need no model downloads:

```bash
python -c "import ast; from pathlib import Path; files=[p for root in ('ProteinLatentLanguage','ProteinLanguageModel') for p in Path(root).rglob('*.py')]; [ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p)) for p in files]; print('Python syntax OK:', len(files))"
python -c "import yaml; from pathlib import Path; files=[p for root in ('ProteinLatentLanguage/configs','ProteinLanguageModel/configs') for p in Path(root).rglob('*.yaml')]; [yaml.safe_load(p.read_text(encoding='utf-8-sig')) for p in files]; print('YAML parsing OK:', len(files))"
```

The YAML check requires PyYAML. Parsing proves syntax, not runtime or numerical correctness. For changed tokenizer or inference behavior, check token ID preservation, BOS/EOS handling, padding, sequence lengths, modality outputs, and relevant CSV columns. Use small fixtures where possible. Check checkpoint tensor compatibility and model forward behavior when the task calls for runtime validation and the dependencies and checkpoint are available.

Review the final diff and run `git diff --check`. Report the files changed, checks run, and remaining limitations. Distinguish syntax checks, CPU behavior checks, GPU execution, and numerical reproduction. Never claim a training or inference result that was not executed.

Commit, push, or publish only when the user requests it. A request to prepare a review draft leaves changes uncommitted.
