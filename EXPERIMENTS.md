# Paper experiments

This guide connects the paper's experiments to the released configurations. Original run mappings and source provenance are recorded in [provenance.json](provenance.json).

## Experiment configurations

| Experiment | Configuration |
| --- | --- |
| PLL Stage 1 reconstruction | [Stage 1](ProteinLatentLanguage/configs/stage1/config.yaml) |
| PLL Stage 2 residue decoding | [Stage 2](ProteinLatentLanguage/configs/stage2/config.yaml) |
| Amino acid PLM scaling | [PLM configurations](ProteinLanguageModel/configs/scaling/aa/) |
| PLLM scaling | [PLLM configurations](ProteinLanguageModel/configs/scaling/pll/) |
| SLLM scaling | [SLLM configurations](ProteinLanguageModel/configs/scaling/sll/) |
| PLM length-conditioned continuation | [PLM continuation](ProteinLanguageModel/configs/continuation/plm/config.yaml) |
| PLLM length-conditioned continuation | [PLLM continuation](ProteinLanguageModel/configs/continuation/pllm/config.yaml) |
| SLLM length-conditioned continuation | [SLLM continuation](ProteinLanguageModel/configs/continuation/sllm/config.yaml) |
| E1-conditioned sequence-to-structure prediction | [Sequence-to-structure configuration](ProteinLanguageModel/configs/sequence_to_structure/config.yaml) |

Each scaling family includes six transformer sizes, with dimension/layer pairs 512/6, 640/12, 768/19, 1024/26, 1536/27, and 2048/34.

SLL tokenizer training is maintained in [vq_encoder_decoder](https://github.com/mahdip72/vq_encoder_decoder). Configurations for the paper's small matched-tokenizer comparison reporting a 34% reduction in best validation perplexity are not included in this repository.

## Training stages

**PLL tokenizer.** Stage 1 learns a contextual latent encoder and codebook by reconstructing frozen ESM-2 150M representations. Stage 2 starts from the best validation checkpoint, freezes the latent encoder and codebook, discards the reconstruction decoder weights, and trains a reinitialized transformer decoder with a linear residue-classification head. Code assignment is deterministic in Stage 2.

**Language-model pretraining.** PLM and PLLM train from scratch for four epochs on amino acid and PLL representations of UniRef50, respectively. SLLM trains from scratch on structure tokens. Its scaling results use the epoch-eight milestone, while the largest saved configuration continues through epoch ten.

**Length-conditioned continuation.** PLM and PLLM continue from their largest models' epoch-four checkpoints for one epoch. SLLM continues from its largest model's epoch-ten checkpoint for two epochs, retaining optimizer state. All three continuation configurations apply length prefixes to 50% of training samples.

**Sequence-to-structure prediction.** Training starts from the SLLM continuation's epoch-two checkpoint, resets optimization, and adds E1-600m sequence context with the last four encoder layers trainable. The reported prediction checkpoint is `checkpoints/steps_64000.pth`, selected within the configured four-epoch training budget.

## Data and tokenizer compatibility

| Corpus | Use | Example data path |
| --- | --- | --- |
| UniRef50 amino acid sequences | PLL tokenizer and PLM | `data/uniref50/{train_set,valid_set,test_set}.csv` |
| PLL-encoded UniRef50 | PLLM | `data/uniref50_pll/{train_set,valid_set}.csv` |
| AFDB structure-token corpus from the GCP-VQVAE pipeline | SLLM pretraining and backbone generation | `data/structure_tokens/{train_set,valid_set}.csv` |
| Paired amino acid sequences and structure codes | E1-conditioned sequence-to-structure training | `data/sequence_structure/{train_set,valid_set}.csv` |

Preserve the original dataset splits and load the saved `tokenizer_vocab.yaml` to retain token IDs and the training alphabet. Structure tokens must be decoded with the checkpoint whose codebook produced the training targets. The larger SLLM generation experiments and the small matched-tokenizer comparison use separate tokenizer lineages.

See the [PLL tokenizer guide](ProteinLatentLanguage/README.md) and [language-model guide](ProteinLanguageModel/README.md) for data formats, training, and inference commands.
