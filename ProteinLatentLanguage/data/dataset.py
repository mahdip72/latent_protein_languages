import random
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


class ProteinDataset(Dataset):
    def __init__(self, csv_file, tokenizer, configs, protein_column_name='Amino Acid Sequence', mode=None):
        """
        Args:
            csv_file (string): Path to the csv file with protein sequences.
            tokenizer: Tokenizer to process protein sequences.
            configs: Configuration object containing dataset paths and other settings.
            protein_column_name (string): The name of the column containing protein sequences.
        """
        # load only the protein column and optionally limit rows for training
        read_kwargs = {'usecols': [protein_column_name], 'low_memory': False}
        if mode == 'train':
            max_samples = configs.train_settings.max_task_samples
            read_kwargs['nrows'] = max_samples
        protein_df = pd.read_csv(csv_file, **read_kwargs)
        seqs = protein_df[protein_column_name].tolist()

        self.protein_sequences = seqs
        self.tokenizer = tokenizer
        self.configs = configs
        self.max_len = configs.model.max_len
        classifier_cfg = getattr(configs.model.vqvae.decoder, 'classifier_head', None)
        self.classifier_head_enabled = bool(getattr(classifier_cfg, 'enabled', False)) if classifier_cfg is not None else False
        self._aa_to_idx = None
        if self.classifier_head_enabled:
            amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
            self._aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}

    def __len__(self):
        return len(self.protein_sequences)

    def _getitem_esm_v2(self, protein_sequence):
        """
        Tokenizes a protein sequence for ESM-v2 model.

        Args:
            protein_sequence (str): The protein amino acid sequence.

        Returns:
            dict: A dictionary containing tokenized 'input_ids' and 'attention_mask'.
        """
        encoded_protein_sequence = self.tokenizer(
            protein_sequence,
            max_length=self.max_len,
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        encoded_protein_sequence['input_ids'] = torch.squeeze(encoded_protein_sequence['input_ids'])
        encoded_protein_sequence['attention_mask'] = torch.squeeze(encoded_protein_sequence['attention_mask'])
        return encoded_protein_sequence

    def _getitem_esm_c(self, protein_sequence):
        """
        Tokenizes a protein sequence for ESM-c model.

        Args:
            protein_sequence (str): The protein amino acid sequence.

        Returns:
            dict: A dictionary containing tokenized 'input_ids' and 'attention_mask'.
        """
        encoded_protein_sequence = self.tokenizer(
            protein_sequence,
            max_length=self.max_len,
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        encoded_protein_sequence['input_ids'] = torch.squeeze(encoded_protein_sequence['input_ids'])
        encoded_protein_sequence['attention_mask'] = torch.squeeze(encoded_protein_sequence['attention_mask'])
        return encoded_protein_sequence

    def __getitem__(self, idx):
        protein_sequence = self.protein_sequences[idx]

        if self.configs.model.protein_encoder.model_type == 'esm_v2':
            encoded_protein_sequence = self._getitem_esm_v2(protein_sequence)
        elif self.configs.model.protein_encoder.model_type == 'esmc':
            encoded_protein_sequence = self._getitem_esm_c(protein_sequence)
        else:
            raise ValueError(f"Unsupported model type: {self.configs.model.protein_encoder.model_type}")

        sample = {'input_ids': encoded_protein_sequence['input_ids'],
                  'attention_mask': encoded_protein_sequence['attention_mask'],
                  'original_sequence': protein_sequence}

        if self.classifier_head_enabled and self._aa_to_idx is not None:
            amino_acids = set(self._aa_to_idx.keys())
            labels = torch.full((self.max_len,), -100, dtype=torch.long)
            seq_upper = ''.join(aa if aa in amino_acids else 'X' for aa in protein_sequence.upper())
            limit = min(len(seq_upper), self.max_len)
            for pos in range(limit):
                labels[pos] = self._aa_to_idx.get(seq_upper[pos], -100)
            sample['classification_labels'] = labels
            sample['original_sequence'] = seq_upper
        else:
            sample['original_sequence'] = protein_sequence.upper()

        return sample


def prepare_dataloaders(configs, logging):
    """
    Prepare the dataloaders for training, validation and test sets.

    Args:
        configs: Configuration object containing dataset paths and other settings.
        logging: Logger object for logging information.

    Returns:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
    """

    if configs.model.protein_encoder.model_type == 'esm_v2':
        # Load the ESM model tokenizer
        tokenizer = AutoTokenizer.from_pretrained(configs.model.protein_encoder.model_name)
    elif configs.model.protein_encoder.model_type == 'esmc':
        from esm.models.esmc import ESMC
        tokenizer = ESMC.from_pretrained(configs.model.protein_encoder.model_name).tokenizer
    else:
        raise ValueError(f"Unsupported model type: {configs.model.protein_encoder.model_type}")

    # Create datasets
    train_dataset = ProteinDataset(csv_file=configs.train_settings.data_path,
                                   tokenizer=tokenizer, configs=configs,
                                   protein_column_name=configs.train_settings.get('protein_column_name', 'Amino Acid Sequence'),
                                   mode='train')
    val_dataset = ProteinDataset(csv_file=configs.valid_settings.data_path,
                                 tokenizer=tokenizer, configs=configs,
                                 protein_column_name=configs.valid_settings.get('protein_column_name', 'Amino Acid Sequence'),
                                 mode='val')
    test_dataset = ProteinDataset(csv_file=configs.test_settings.data_path,
                                  tokenizer=tokenizer, configs=configs,
                                  protein_column_name=configs.test_settings.get('protein_column_name', 'Amino Acid Sequence'),
                                  mode='test')

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=configs.train_settings.batch_size,
                                               num_workers=configs.train_settings.num_workers,
                                               pin_memory=configs.train_settings.pin_memory,
                                               persistent_workers=configs.train_settings.persistent_workers,
                                               shuffle=configs.train_settings.shuffle,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=configs.valid_settings.batch_size,
                                             num_workers=configs.valid_settings.num_workers,
                                             pin_memory=configs.valid_settings.pin_memory,
                                             persistent_workers=configs.valid_settings.persistent_workers,
                                             drop_last=True,
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=configs.test_settings.batch_size,
                                              num_workers=configs.test_settings.num_workers,
                                              pin_memory=False,
                                              persistent_workers=False,
                                              drop_last=True,
                                              shuffle=False)

    logging.info("Dataloaders prepared successfully.")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import logging
    import yaml
    from utils.utils import load_configs

    # Load configuration from a YAML file
    with open('../configs/config.yaml', 'r') as file:
        test_configs = yaml.safe_load(file)

    # Convert to Box for easier access
    test_configs = load_configs(test_configs)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Prepare dataloaders
    test_train_loader, test_val_loader, test_test_dataloader = prepare_dataloaders(test_configs, logging)

    # Example usage of the dataloaders
    for batch in test_train_loader:
        print(batch['original_sequence'])  # Print original protein sequences
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        break  # Remove this to iterate through the entire dataset
