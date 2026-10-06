# Experiment and source provenance

## Source selection

The saved run directories were inspected in the archived ProteinLatentLanguage and ProteinLanguageModel projects. The newer H: research projects were not used as the release source.

- PLL source is recovered from archived Git revision `299e14419770`, dated October 24, 2025. It has the transformer decoder required by the original Stage 2 checkpoint. The later MLP implementation in the archive's working tree was excluded.
- Language-model source comes from archived tracked revision `9e1720e6adb5`, dated May 4, 2026. Mutable working-tree configs were excluded. Training configs and vocabularies come directly from saved results. Compatibility changes to this code are listed in `provenance.json`.
- The saved runs did not record a source commit hash. These are identified recoverable source revisions, not a claim that each run contains an exact source snapshot. An exact original dependency lockfile was also not available.

The original PLL Stage 2 checkpoint was inspected without loading its tensor data. Its decoder input projection is `[768, 128]`, its residue head is `[21, 768]`, and its transformer has 122 attention-layer tensors. The Stage 2 log reports validation residue accuracy `0.9980847835540771`, which rounds to 99.81%.

## Saved training runs

Run paths below are relative to the named archived project. Dates are encoded in their run names. Public configs preserve the training parameters while replacing data/output locations with portable example paths. The checkpoint filenames recorded in `provenance.json` describe what was present in the archive, not bundled model downloads.

| Project | Role | Saved run | Release configuration |
| --- | --- | --- | --- |
| ProteinLatentLanguage | PLL Stage 1 reconstruction | `results/final/2025-10-21__16-31-28` | [config](ProteinLatentLanguage/configs/stage1/config.yaml) |
| ProteinLatentLanguage | PLL Stage 2 residue decoding | `results/final/stage_2/2025-10-24__21-30-50` | [config](ProteinLatentLanguage/configs/stage2/config.yaml) |
| ProteinLanguageModel | aa scaling model | `results/amino_acid/2025-10-25__21-59-08` | [config](ProteinLanguageModel/configs/scaling/aa/d512_l6/config.yaml) |
| ProteinLanguageModel | aa scaling model | `results/amino_acid/2025-10-26__10-34-30` | [config](ProteinLanguageModel/configs/scaling/aa/d640_l12/config.yaml) |
| ProteinLanguageModel | aa scaling model | `results/amino_acid/2025-10-27__13-03-38` | [config](ProteinLanguageModel/configs/scaling/aa/d768_l19/config.yaml) |
| ProteinLanguageModel | aa scaling model | `results/amino_acid/2025-10-29__13-34-48` | [config](ProteinLanguageModel/configs/scaling/aa/d1024_l26/config.yaml) |
| ProteinLanguageModel | aa scaling model | `results/amino_acid/2025-11-22__12-26-17` | [config](ProteinLanguageModel/configs/scaling/aa/d1536_l27/config.yaml) |
| ProteinLanguageModel | aa scaling model | `results/amino_acid/2025-11-27__07-49-39` | [config](ProteinLanguageModel/configs/scaling/aa/d2048_l34/config.yaml) |
| ProteinLanguageModel | pll scaling model | `results/pll_esm150/2025-11-02__01-00-06` | [config](ProteinLanguageModel/configs/scaling/pll/d512_l6/config.yaml) |
| ProteinLanguageModel | pll scaling model | `results/pll_esm150/2025-11-02__13-54-07` | [config](ProteinLanguageModel/configs/scaling/pll/d640_l12/config.yaml) |
| ProteinLanguageModel | pll scaling model | `results/pll_esm150/2025-11-03__16-03-32` | [config](ProteinLanguageModel/configs/scaling/pll/d768_l19/config.yaml) |
| ProteinLanguageModel | pll scaling model | `results/pll_esm150/2025-11-05__16-52-59` | [config](ProteinLanguageModel/configs/scaling/pll/d1024_l26/config.yaml) |
| ProteinLanguageModel | pll scaling model | `results/pll_esm150/2025-11-09__01-27-58` | [config](ProteinLanguageModel/configs/scaling/pll/d1536_l27/config.yaml) |
| ProteinLanguageModel | pll scaling model | `results/pll_esm150/2025-11-13__21-24-14` | [config](ProteinLanguageModel/configs/scaling/pll/d2048_l34/config.yaml) |
| ProteinLanguageModel | sll scaling model | `results/structure/2026-01-18__21-41-29` | [config](ProteinLanguageModel/configs/scaling/sll/d640_l12/config.yaml) |
| ProteinLanguageModel | sll scaling model | `results/structure/2026-01-19__21-16-32` | [config](ProteinLanguageModel/configs/scaling/sll/d768_l19/config.yaml) |
| ProteinLanguageModel | sll scaling model | `results/structure/2026-01-21__14-09-34` | [config](ProteinLanguageModel/configs/scaling/sll/d1024_l26/config.yaml) |
| ProteinLanguageModel | sll scaling model | `results/structure/2026-01-24__11-35-02` | [config](ProteinLanguageModel/configs/scaling/sll/d1536_l27/config.yaml) |
| ProteinLanguageModel | sll scaling model | `results/structure/2026-01-29__01-10-22` | [config](ProteinLanguageModel/configs/scaling/sll/d2048_l34/config.yaml) |
| ProteinLanguageModel | sll scaling model | `results/structure/2026-02-08__04-31-14` | [config](ProteinLanguageModel/configs/scaling/sll/d512_l6/config.yaml) |
| ProteinLanguageModel | plm length-conditioned continuation | `results/ablation/condition/amino_acid/2025-12-24__23-09-58` | [config](ProteinLanguageModel/configs/continuation/plm/config.yaml) |
| ProteinLanguageModel | pllm length-conditioned continuation | `results/ablation/condition/pll_esm150/2025-12-27__13-52-03` | [config](ProteinLanguageModel/configs/continuation/pllm/config.yaml) |
| ProteinLanguageModel | sllm length-conditioned continuation | `results/structure/2026-02-09__15-50-09` | [config](ProteinLanguageModel/configs/continuation/sllm/config.yaml) |
| ProteinLanguageModel | E1-conditioned sequence-to-structure training | `results/sequence_to_structure/2026-03-21__15-50-45` | [config](ProteinLanguageModel/configs/sequence_to_structure/config.yaml) |

## Training lineage

PLL Stage 1 uses frozen ESM-2 150M features and trains a contextual latent encoder/codebook with representation reconstruction. Stage 2 resumes its best validation checkpoint, freezes the latent encoder and codebook, discards the reconstruction decoder weights, and trains a new transformer residue decoder. Its quantizer configuration disables stochastic sampling.

The six AA and six PLL scaling runs start from scratch and use four-epoch saved training budgets. Their content vocabulary differs, so their embedding parameter counts differ. The first small PLL config has a sample cap of 80 million and most other sequence configs have a cap of 100 million. These are upper bounds, not proof that this many samples were present in the selected data files.

The six SLL scaling runs start from scratch on structure tokens. Their configs retain inactive resume paths even when `resume.enable` is false. The erroneous `structure/wrong_validation/2026-01-18__11-04-20` run was excluded. The corrected small model is `2026-02-08__04-31-14`.

The largest structure run, `2026-01-29__01-10-22`, is configured for ten epochs. The scaling milestone was epoch eight, when length conditioning was scheduled to become active. That run continued to epoch ten. The follow-up run `2026-02-09__15-50-09` resumed epoch ten, kept optimizer state, and trained two more epochs. The saved continuation config applies length conditioning with probability 0.5.

The selected AA and PLL length-conditioned continuations start from the corresponding largest model's epoch-four checkpoint. Their saved configs specify one epoch and a length-prefix probability of 0.5. The local continuation directories contain config/vocabulary files but no `.pth` files, so the release inference examples use editable checkpoint placeholders for those runs. Their final sampling checkpoint filenames cannot be confirmed from these two local result trees alone.

The E1-conditioned sequence-to-structure run `2026-03-21__15-50-45` starts from the SLLM continuation's epoch-two checkpoint, resets optimization, and connects the E1-600m sequence encoder with its last four layers trainable. The reported prediction checkpoint is `checkpoints/steps_64000.pth`. The experiment notes record 32 GPUs, a per-GPU batch of eight, and 4,315,655,500 unmasked tokens processed at that checkpoint. The saved four-epoch budget is longer than this reported milestone.

## Data and tokenizer lineage

| Corpus | Use | Release path convention |
| --- | --- | --- |
| UniRef50 amino acid sequences | PLL tokenizer and AA language model | `data/uniref50/{train_set,valid_set,test_set}.csv` |
| PLL-encoded UniRef50 | PLL language model | `data/uniref50_pll/{train_set,valid_set}.csv` |
| AFDB structure-token corpus from the GCP-VQVAE pipeline | SLLM pretraining and backbone generation | `data/structure_tokens/{train_set,valid_set}.csv` |
| Paired amino acid sequences and structure codes | E1-conditioned sequence-to-structure continuation | `data/sequence_structure/{train_set,valid_set}.csv` |

The original local dataset paths and checkpoint files are recorded in the private preparation audit. Dataset CSVs were not copied into this release. Preserve the original split membership when reproducing a reported run.

The original sequence models' saved vocabularies contain the 20 canonical amino acids. X belongs to the PLL decoder's 21-class residue classifier, while later language-model code added X to its default AA vocabulary. Release training and inference explicitly load saved vocabulary exports so this later change does not alter historical token IDs or embedding sizes.

Structure token IDs only have meaning relative to their trained codebook. The large structure generation and E1 sequence-to-structure records refer to the Lite decoder run `2025-12-30__18-09-47`. This is a distinct lineage from the matched small-data tokenizer comparison. Keep its decoder and target tokenization together.

## Separate tokenizer experiments

SLL/GCP-VQVAE 2 training and source code are maintained in [vq_encoder_decoder](https://github.com/mahdip72/vq_encoder_decoder), as a separate release. They are intentionally not copied here.

The manuscript's 34% matched-tokenizer comparison is recorded as Prot2Token-style sequence-to-structure runs `old_lite/2026-04-29__19-37-26` and `new_lite/2026-04-30__12-19-56`, using an ESM-2 650M encoder and a 12-layer, dimension-1280 decoder. Those original training configurations/source snapshots were not located in the two requested archives. This release does not label the unrelated April 28 unconditional `ablation_structure/old_lite` and `new_lite` configs as those runs. The April 28 runs use a dimension-768, 19-layer unconditional structure model and are outside the released main experiment set.

## Release adaptations

- PLL loads and saves learnable.yaml beside the selected stage config, preserving stochastic Stage 1 versus deterministic Stage 2.
- PLL checkpoint loading explicitly uses the historical weights_only=False behavior for trusted training checkpoints.
- Removed the optional generation-length plotting helper and call. Generation outputs and training are unchanged.
- Language-model training loads the saved vocabulary beside the selected config. Inference loads the vocabulary saved with its checkpoint. This preserves the historical AA alphabet and token IDs.
- Standalone E1-conditioned inference now passes the same sequence-context tensors to generation and token scoring as the archived evaluation path.
- Token scoring unwraps the model before accessing the transformer in distributed inference.
- DeepConf confidence/consensus functions were extracted from the archived evaluator into an inference-only module. No ProteinBench or refolding code was copied.
- The later unmasked attention optimization is opt-in and off for saved paper configs, preserving the historical masked training path.

## Validation

The preparation checks cover Python syntax, YAML parsing, internal import closure, all 24 saved training configs, all 22 saved language-model vocabularies, the PLL CSV adapter, and entropy/consensus selection behavior. Vocabulary construction was checked using pure Python without tensor operations. The original PLL decoder dimensions were checked against checkpoint metadata.

GPU training, model forward passes, pretrained-model downloads, and coordinate decoding were not run during packaging. Numerical reproduction still requires the original datasets, trained checkpoints, and compatible dependency versions.
