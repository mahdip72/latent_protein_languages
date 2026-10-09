# Autoregressive protein language models

This folder trains causal transformers with a next-token prediction objective on amino acids (PLM), PLL codes (PLLM), or structure codes (SLLM). The pretrained sequence and structure models are separate. The sequence-to-structure continuation connects E1 sequence representations to the structure language model.

## Environment

Install a CUDA-enabled PyTorch build, then:

```bash
cd ProteinLanguageModel
python -m pip install -r requirements.txt
```

E1-conditioned training and inference also require the `E1` package providing `E1.modeling.E1Model` and `E1.tokenizer.get_tokenizer`. Follow the [upstream E1 installation instructions](https://github.com/Profluent-AI/E1#installation) in the same environment. The paper configuration selects `Profluent-Bio/E1-600m` and fine-tunes its last four layers.

The archived dependencies were not saved as a complete environment lockfile. bf16 and 8-bit Adam are the recorded training defaults. PyArrow supports the archived large-CSV loading path. The original loader may create a sibling `chunks/` directory containing training CSV shards.

## Data

Provide training and validation CSVs, then update the selected config's `data_path` values.

| Task | Accepted content columns | Content format |
| --- | --- | --- |
| Amino acid PLM | `Amino Acid Sequence` or `amino_acid_sequence` | Amino acid letters |
| PLLM | `indices` or `pll` | Whitespace-separated integer PLL codes |
| SLLM | `structures` or `structure` | Whitespace-separated structure codes |
| E1 sequence-to-structure | An amino acid column and a structure column from above | Aligned sequence and structure targets |

PLL and structure codes range from 0 to 4095. Missing structure residues use -1 in the archived protocol. The tokenizer assigns the structure modality its own `_3D` vocabulary entries. Supply raw integer code strings in the CSV, not BOS/EOS or condition-token strings. Preserve the tokenizer and decoder checkpoint used to create those codes.

## Tokenization policy

Following Appendix A.5 of the [paper](https://arxiv.org/abs/2610.03978), each valid residue contributes one content token: an amino acid letter, a PLL code, or an SLL code. PLL and SLL each use 4,096 code states. Unconditional streams use:

```text
<BOS> content_tokens... <EOS>
```

Length conditioning adds `<LEN_n>`, where `n` is the content length after truncation. Conditional PLL streams use `<BOP>` / `<EOP>` boundaries, and conditional SLL streams use `<BO3D>` / `<EO3D>`. For E1-conditioned sequence-to-structure prediction, E1 encodes the sequence separately as model context, and the autoregressive target stream is:

```text
<BOS> <sequence_to_structure> <BO3D> SLL_tokens... <EO3D> <EOS>
```

Training uses next-token prediction. Streams are padded to the model context length with `<PAD>`. Padding and condition-prefix targets are excluded from loss and perplexity, while content and boundary tokens are predicted normally. Load the run's saved `tokenizer_vocab.yaml` to preserve its token IDs and amino acid alphabet.

## Pretrain

The [scaling configs](configs/scaling/) contain the six recorded sizes for each of the three modalities. For example:

```bash
accelerate launch --num_processes 1 train.py --config_path configs/scaling/aa/d512_l6/config.yaml
accelerate launch --num_processes 1 train.py --config_path configs/scaling/pll/d512_l6/config.yaml
accelerate launch --num_processes 1 train.py --config_path configs/scaling/sll/d512_l6/config.yaml
```

The full grid uses dimensions/layers 512/6, 640/12, 768/19, 1024/26, 1536/27, and 2048/34. The saved AA and PLL runs train for four epochs. The structure runs use eight epochs, with the largest saved config continuing through epoch ten. The SLL scaling milestone is epoch eight. Read [EXPERIMENTS.md](../EXPERIMENTS.md) for experiment configurations and training stages.

Use `--multi_gpu --num_processes N` for distributed training. The configs specify per-process batches, so preserve the paper's global batch by accounting for GPU count and gradient accumulation. Each run saves its training config, tokenizer vocabulary, logs, and checkpoints. Keep `config.yaml` and `tokenizer_vocab.yaml` with the checkpoint directory when moving trained models.

## Continue with length conditioning

The [continuation configs](configs/continuation/) contain the saved PLM, PLLM, and SLLM length-conditioned runs. Point `resume.resume_path` at the appropriate pretrained checkpoint:

```bash
accelerate launch --num_processes 1 train.py --config_path configs/continuation/pllm/config.yaml
```

Use `plm` or `sllm` for the other modalities. The saved configs apply the length prefix with probability 0.5. The SLL continuation resumes the largest structure model's epoch-ten checkpoint, enables length conditioning from its first epoch, and continues for two epochs. Its config retains the optimizer state.

## Generate sequences or backbones

Set `trained_model_dir` and the relative `checkpoint_path` in the chosen [inference config](configs/inference/). Then:

```bash
python inference_generate_de_novo.py --config_path configs/inference/pllm_unconditional.yaml
python inference_generate_de_novo.py --config_path configs/inference/plm_length_conditioned.yaml
python inference_generate_de_novo.py --config_path configs/inference/sllm_length_conditioned.yaml
```

The unconditional examples generate 500 unfiltered samples with temperature 1.0 and nucleus probability 0.9. They do not require EOS and do not apply entropy filtering. For a temperature sweep, vary `sampling.temperature` while preserving the other settings. These are explicit starting examples, not a bundled reproduction of every evaluation run.

The length-conditioned examples set `fixed_value: 100`. Change this value for other target lengths. The sequence examples use temperature 0.5 and the structure example uses 1.0. This repository provides sampling, not the ProteinBench filtering/refolding pipeline.

AA outputs contain `amino_acid_sequence`. PLL outputs contain `pll_sequence`, which must be decoded to amino acids using the [PLL decoder](../ProteinLatentLanguage/README.md#decode-pll-tokens). Structure outputs contain `structure_sequence`.

### Decode structure tokens

The GCP-VQVAE implementation is external: [vq_encoder_decoder](https://github.com/mahdip72/vq_encoder_decoder). Its inference package provides `gcp_vqvae.GCPVQVAE`. Install the standalone package as described in the [upstream guide](https://github.com/mahdip72/vq_encoder_decoder#standalone-python-package-gcp-vqvae):

```bash
python -m pip install "git+https://github.com/mahdip72/vq_encoder_decoder.git@master#subdirectory=gcp-vqvae"
```

Automatic backbone decoding is off in the examples until a compatible tokenizer run is supplied. To enable it, set `vqvae_decode.enabled: true` and `vqvae_decode.trained_model_dir` to a directory containing:

```text
config_vqvae.yaml
config_gcpnet_encoder.yaml
config_geometric_decoder.yaml
checkpoints/best_valid.pth
```

Use the decoder trained with the same codebook as the model's targets. The large structure-generation/sequence-to-structure lineage points to the archived Lite run `2025-12-30__18-09-47`. Do not substitute a new SLL codebook just because both vocabularies have 4,096 entries.

## Train sequence-to-structure

[The saved E1 configuration](configs/sequence_to_structure/config.yaml) starts from the SLLM continuation's epoch-two checkpoint. It enables E1 sequence context and predicts structure tokens with next-token supervision.

```bash
accelerate launch --num_processes 1 train.py --config_path configs/sequence_to_structure/config.yaml
```

The paper used `checkpoints/steps_64000.pth` from run `2026-03-21__15-50-45`. The saved config specifies four epochs, while the reported experiment is the step-64,000 checkpoint. Do not equate that checkpoint with the final epoch of the configured budget.

## Predict structures

Set the model directory, checkpoint, and input CSV in [sequence_to_structure.yaml](configs/inference/sequence_to_structure.yaml):

```bash
python inference_sequence_to_structure.py --config_path configs/inference/sequence_to_structure.yaml
```

This example uses greedy decoding. [sequence_to_structure_samples.yaml](configs/inference/sequence_to_structure_samples.yaml) is an editable stochastic example with 64 trajectories per sequence and temperature 1.0. Both request raw and temperature-adjusted token probabilities and entropies. The examples emit token predictions until a compatible GCP-VQVAE decoder is configured.

The main prediction path uses E1 with amino acid inputs. The optional older PLL-input branch is not used by this paper configuration and would require the separate `pll` Python wrapper. For paper inference, leave `use_pll: false` as recorded.

## Select a sampled structure with internal confidence

The [inference selector](inference_select_structures.py) scores raw token entropy in overlapping windows, retains the most confident candidates, and selects the token-consensus representative. Its core functions are taken from the archived DeepConf evaluation path, without the refolding or benchmark pipeline. The [selection config](configs/inference/confidence_selection.yaml) uses a window of 256, stride 1, the worst 10% of windows, and consensus within the most confident half of candidates.

```bash
python inference_select_structures.py --input path/to/predicted_structures.csv --output selected_structures.csv --config_path configs/inference/confidence_selection.yaml
```

Use multi-sample prediction output with `prediction_id`, `predicted_structures`, and `predicted_structure_token_entropy_raw` columns. The script keeps one generated token trajectory per input. Decode those selected tokens with the same external GCP-VQVAE checkpoint. Selection uses model confidence and token agreement, without reference structures or refolding quality scores.

The supplied training configs and checkpoint-based inference load the corresponding saved `tokenizer_vocab.yaml` automatically. This preserves the paper runs' token IDs and their 20-residue amino acid alphabet despite the later tokenizer code's default X class.
