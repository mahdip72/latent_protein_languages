"""
Protein sequence tokenizer with extensible conditioning support.

Conditions are treated as data (a list of Condition objects), not code branches.
Adding a new condition requires only adding an entry to CONDITION_REGISTRY.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

import torch
import yaml


# =============================================================================
# Condition Definition
# =============================================================================

@dataclass
class Condition:
    """
    A single conditioning type for protein sequence tokenization.
    
    Each condition can optionally:
    - Add a prefix token to the sequence (e.g., <C2N>, <LEN_X>)
    - Transform the content tokens (e.g., reverse the sequence)
    
    Conditions are stored in CONDITION_REGISTRY and instantiated by the tokenizer
    based on the config. During encoding, conditions are evaluated per-sample
    to support classifier-free guidance (CFG) training where some samples are
    conditioned and others are not.
    
    Attributes:
        name (str): Identifier matching the config key (e.g., 'length', 'c2n').
            Used to look up settings in condition_config dict.
        token (str | Callable | None): The prefix token to add when this condition fires.
            - str: Static token (e.g., '<C2N>')
            - Callable[[int], str]: Dynamic token based on content length (e.g., '<LEN_5>')
            - None: No token added (condition only applies transform)
        transform (Callable | None): Optional function to transform content tokens.
            Signature: (List[str]) -> List[str]. Applied before truncation.
        order (int): Controls prefix token ordering. Lower values appear first.
            Example: c2n (order=0) appears before length (order=1) → [<C2N>, <LEN_5>]
        enabled (bool): Runtime flag set by tokenizer based on config.
        prob (float): Probability of firing when not forced. Set from config.
    
    Example:
        # Define a new condition in CONDITION_REGISTRY:
        'my_cond': Condition(
            name='my_cond',
            token='<MY_TOKEN>',           # or callable for dynamic tokens
            transform=my_transform_fn,     # or None
            order=2,                       # appears after c2n and length
        )
        
        # Then in config.yaml:
        condition_tokens:
          my_cond:
            enabled: True
            probability: 0.5
    """
    name: str
    token: Optional[str | Callable[[int], str]] = None
    transform: Optional[Callable[[List[str]], List[str]]] = None
    order: int = 0
    
    # Runtime state (set during encoding based on config)
    enabled: bool = field(default=False, repr=False)
    prob: float = field(default=0.0, repr=False)
    
    def should_fire(self, forced: Optional[bool] = None) -> bool:
        """
        Determine if this condition should be applied for a given sample.
        
        Supports three modes:
        1. Condition disabled (enabled=False): Always returns False
        2. Forced override: Returns the forced value directly
        3. Random sampling: Returns True with probability `self.prob`
        
        This enables classifier-free guidance (CFG) training where the model
        sees a mix of conditioned and unconditioned samples.
        
        Args:
            forced: Override the random decision.
                - None: Use random sampling based on self.prob
                - True: Force condition to fire
                - False: Force condition to not fire
        
        Returns:
            bool: Whether this condition should be applied to the sample.
        
        Example:
            >>> cond = Condition(name='length', enabled=True, prob=0.5)
            >>> cond.should_fire()        # Random: ~50% True
            >>> cond.should_fire(True)    # Always True
            >>> cond.should_fire(False)   # Always False
            
            >>> cond_disabled = Condition(name='length', enabled=False, prob=1.0)
            >>> cond_disabled.should_fire()  # Always False (disabled)
        """
        if not self.enabled:
            return False
        if forced is not None:
            return forced
        return random.random() < self.prob
    
    def get_token(self, content_length: int) -> Optional[str]:
        """
        Get the prefix token string for this condition.
        
        Handles both static tokens (strings) and dynamic tokens (callables).
        Dynamic tokens are useful for conditions like length where the token
        depends on the actual content (e.g., '<LEN_5>' for a 5-residue sequence).
        
        Args:
            content_length: The length of the content tokens (after truncation).
                Used by dynamic token functions to generate length-specific tokens.
        
        Returns:
            The token string to add to the sequence, or None if this condition
            has no associated token (transform-only condition).
        
        Example:
            # Static token (C2N condition):
            >>> c2n = Condition(name='c2n', token='<C2N>')
            >>> c2n.get_token(10)
            '<C2N>'
            
            # Dynamic token (length condition):
            >>> length = Condition(name='length', token=lambda n: f'<LEN_{n}>')
            >>> length.get_token(5)
            '<LEN_5>'
            >>> length.get_token(100)
            '<LEN_100>'
            
            # No token (transform-only):
            >>> transform_only = Condition(name='custom', token=None, transform=some_fn)
            >>> transform_only.get_token(10)
            None
        """
        if self.token is None:
            return None
        if callable(self.token):
            return self.token(content_length)
        return self.token
    
    def apply_transform(self, tokens: List[str]) -> List[str]:
        """
        Apply this condition's transform to the content tokens.
        
        Transforms modify the content sequence itself, not just add prefix tokens.
        For example, the C2N condition reverses the sequence to represent
        C-terminal to N-terminal generation direction.
        
        Transforms are applied in condition order before truncation, so the
        transformation sees the full sequence. Multiple conditions' transforms
        are applied sequentially based on their `order` attribute.
        
        Args:
            tokens: List of content tokens (amino acids or PLL indices).
                These are the raw content tokens before any special tokens
                (BOS, BOP, etc.) are added.
        
        Returns:
            Transformed token list. Returns input unchanged if no transform defined.
        
        Example:
            # C2N condition reverses sequence:
            >>> c2n = Condition(
            ...     name='c2n',
            ...     token='<C2N>',
            ...     transform=lambda t: list(reversed(t))
            ... )
            >>> c2n.apply_transform(['M', 'D', 'E', 'A', 'A'])
            ['A', 'A', 'E', 'D', 'M']
            
            # Condition with no transform:
            >>> length = Condition(name='length', token='<LEN_5>', transform=None)
            >>> length.apply_transform(['M', 'D', 'E'])
            ['M', 'D', 'E']  # Unchanged
            
            # Custom transform (hypothetical masking condition):
            >>> def mask_cysteines(tokens):
            ...     return ['<MASK>' if t == 'C' else t for t in tokens]
            >>> mask_cond = Condition(name='mask_c', transform=mask_cysteines)
            >>> mask_cond.apply_transform(['M', 'C', 'A', 'C', 'K'])
            ['M', '<MASK>', 'A', '<MASK>', 'K']
        """
        if self.transform is None:
            return tokens
        return self.transform(tokens)


# =============================================================================
# Condition List (ordered) - add new conditions by appending
# =============================================================================

def _make_length_token(length: int) -> str:
    """Generate length token string."""
    return f'<LEN_{length}>'


def _reverse_tokens(tokens: List[str]) -> List[str]:
    """Reverse token sequence (C→N terminal)."""
    return list(reversed(tokens))


# Ordered list of supported conditions. Add new ones here.
BASE_CONDITIONS: List[Condition] = [
    Condition(
        name='sequence_to_structure',
        token='<sequence_to_structure>',
        transform=None,
        order=0,
    ),
    Condition(
        name='structure_to_sequence',
        token='<structure_to_sequence>',
        transform=None,
        order=1,
    ),
    Condition(
        name='c2n',
        token='<C2N>',
        transform=_reverse_tokens,
        order=2,  # kept for clarity, but list order wins
    ),
    Condition(
        name='length',
        token=_make_length_token,  # Dynamic token based on content length
        transform=None,
        order=3,
    ),
]


# =============================================================================
# Tokenizer
# =============================================================================

class ProteinTokenizer:
    """
    Tokenizer for protein sequences with extensible conditioning support.
    
    Supports two modes:
    
    1. **Standard mode** (no conditioning enabled):
       - Sequence: [BOS, content..., EOS]
       - Max content: max_len - 1
    
    2. **Conditioning mode** (any condition enabled):
       - Adds <BOP>/<EOP> markers around protein content
       - Condition tokens appear between BOS and BOP (in order)
       - Sequence: [BOS, <cond_tokens>..., <BOP>, content..., <EOP>, EOS]
       - Max content: max_len - 3 - num_active_condition_tokens
    
    Args:
        data_type: 'amino_acid', 'pll', or 'structure'
        condition_config: Dict like {'length': {'enabled': True, 'probability': 0.5}, ...}
        max_len: Required when any conditioning is enabled
    """
    
    # Base special tokens (always present)
    BASE_SPECIAL_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<MASK>', '<UNK>']
    STRUCTURE_TOKEN_COUNT = 4096
    MISSING_STRUCTURE_TOKEN = '-1_3D'
    
    def __init__(
        self,
        data_type: str,
        condition_config: Optional[Dict[str, Any]] = None,
        max_len: Optional[int] = None,
        vocab_path: Optional[str] = None,
    ):
        self.data_type = data_type
        self.max_len = max_len
        
        # Initialize conditions from ordered list
        self.conditions: List[Condition] = []
        self._init_conditions(condition_config, max_len)
        
        # Check if any conditioning is active
        self.conditioning_enabled = any(c.enabled for c in self.conditions)
        self.sequence_to_structure_enabled = any(
            c.enabled and c.name == 'sequence_to_structure' for c in self.conditions
        )
        self.structure_to_sequence_enabled = any(
            c.enabled and c.name == 'structure_to_sequence' for c in self.conditions
        )
        self.structure_pair_enabled = (
            self.sequence_to_structure_enabled or self.structure_to_sequence_enabled
        )
        self._init_pair_condition_settings(condition_config)
        
        # Build vocabulary
        self._build_vocabulary(data_type, max_len)
        if vocab_path is not None:
            with open(vocab_path, encoding='utf-8') as handle:
                saved = yaml.safe_load(handle)
            mapping = saved['token_to_id']
            if sorted(mapping.values()) != list(range(len(mapping))):
                raise ValueError('Saved tokenizer vocabulary must have contiguous, unique IDs.')
            if not self._special_tokens.issubset(mapping):
                raise ValueError('Saved vocabulary is missing required special/condition tokens.')
            self.token_to_id = dict(mapping)
            self.id_to_token = {index: token for token, index in mapping.items()}
            self.tokenizer_vocab_size = len(mapping)
        
        # Store frequently used token IDs
        self.pad_token_id = self.token_to_id['<PAD>']
        self.bos_token_id = self.token_to_id['<BOS>']
        self.eos_token_id = self.token_to_id['<EOS>']
        self.mask_token_id = self.token_to_id['<MASK>']
        self.unk_token_id = self.token_to_id['<UNK>']
    
    def _init_conditions(self, config: Optional[Dict], max_len: Optional[int]) -> None:
        """Initialize condition objects from config."""
        if not config:
            return
        
        for cond_template in BASE_CONDITIONS:
            name = cond_template.name
            cond_cfg = config.get(name, {})
            if cond_cfg.get('enabled', False):
                if max_len is None:
                    raise ValueError(f"max_len required when {name} conditioning is enabled")
                
                if 'probability' not in cond_cfg:
                    raise ValueError(f"condition_tokens.{name}.probability is required when enabled")
                # Create a copy with runtime state
                cond = Condition(
                    name=cond_template.name,
                    token=cond_template.token,
                    transform=cond_template.transform,
                    order=cond_template.order,
                    enabled=True,
                    prob=float(cond_cfg['probability']),
                )
                self.conditions.append(cond)
        
        # List order already defines prefix order; keep as-is

    def _init_pair_condition_settings(self, condition_config: Optional[Dict]) -> None:
        """Capture per-condition sequence modality settings for pairing conditions."""
        self.sequence_to_structure_use_pll = None
        self.structure_to_sequence_use_pll = None
        self.sequence_to_structure_sequence_modality = None
        self.structure_to_sequence_sequence_modality = None
        self.sequence_to_structure_protein_encoder_context_cfg = {}
        self.sequence_to_structure_use_protein_encoder_context = False
        self.sequence_to_structure_missing_residue_mapping = False
        self.structure_to_sequence_missing_residue_mapping = False
        self.sequence_to_structure_augmentation = {'probability': 0.0, 'max_percentage': 0.0}
        self.structure_to_sequence_augmentation = {'probability': 0.0, 'max_percentage': 0.0}

        if not condition_config:
            return
        if hasattr(condition_config, 'to_dict'):
            condition_config = condition_config.to_dict()

        if self.sequence_to_structure_enabled:
            seq_cfg = condition_config.get('sequence_to_structure', {})
            context_cfg = seq_cfg.get('protein_encoder_context', {})
            if hasattr(context_cfg, 'to_dict'):
                context_cfg = context_cfg.to_dict()
            context_cfg = dict(context_cfg) if context_cfg else {}
            context_enabled = bool(
                context_cfg.get('enable', context_cfg.get('enabled', False))
            )
            self.sequence_to_structure_protein_encoder_context_cfg = context_cfg
            self.sequence_to_structure_use_protein_encoder_context = context_enabled

            if context_enabled:
                self.sequence_to_structure_use_pll = False
            else:
                if 'use_pll' not in seq_cfg:
                    raise ValueError("condition_tokens.sequence_to_structure.use_pll is required when enabled")
                self.sequence_to_structure_use_pll = bool(seq_cfg['use_pll'])
            self.sequence_to_structure_sequence_modality = (
                'pll' if self.sequence_to_structure_use_pll else 'amino_acid'
            )
            self.sequence_to_structure_missing_residue_mapping = bool(
                seq_cfg.get('map_missing_residue_tokens', False)
            )
            self.sequence_to_structure_augmentation = self._normalize_augmentation_cfg(
                seq_cfg.get('augmentation', {})
            )
        if self.structure_to_sequence_enabled:
            struct_cfg = condition_config.get('structure_to_sequence', {})
            if 'use_pll' not in struct_cfg:
                raise ValueError("condition_tokens.structure_to_sequence.use_pll is required when enabled")
            self.structure_to_sequence_use_pll = bool(struct_cfg['use_pll'])
            self.structure_to_sequence_sequence_modality = (
                'pll' if self.structure_to_sequence_use_pll else 'amino_acid'
            )
            self.structure_to_sequence_missing_residue_mapping = bool(
                struct_cfg.get('map_missing_residue_tokens', False)
            )
            self.structure_to_sequence_augmentation = self._normalize_augmentation_cfg(
                struct_cfg.get('augmentation', {})
            )
    
    def _build_vocabulary(self, data_type: str, max_len: Optional[int]) -> None:
        """Build the token vocabulary."""
        if data_type not in ('amino_acid', 'pll', 'structure'):
            raise ValueError(f"Unknown data_type: {data_type}")

        require_amino_acid = data_type == 'amino_acid'
        require_pll = data_type == 'pll'
        require_structure = data_type == 'structure' or self.structure_pair_enabled

        if (
            self.sequence_to_structure_sequence_modality == 'amino_acid'
            and not self.sequence_to_structure_use_protein_encoder_context
        ):
            require_amino_acid = True
        elif self.sequence_to_structure_sequence_modality == 'pll':
            require_pll = True
        if self.structure_to_sequence_sequence_modality == 'amino_acid':
            require_amino_acid = True
        elif self.structure_to_sequence_sequence_modality == 'pll':
            require_pll = True

        amino_acid_tokens = list("ACDEFGHIKLMNPQRSTVWYX")
        pll_tokens = [str(i) for i in range(4096)]
        structure_tokens = [self.MISSING_STRUCTURE_TOKEN] + [
            f"{i}_3D" for i in range(self.STRUCTURE_TOKEN_COUNT)
        ]

        content_tokens: list[str] = []
        seen: set[str] = set()

        def add_tokens(tokens: list[str]) -> None:
            for tok in tokens:
                if tok not in seen:
                    content_tokens.append(tok)
                    seen.add(tok)

        if data_type == 'amino_acid':
            add_tokens(amino_acid_tokens)
        elif data_type == 'pll':
            add_tokens(pll_tokens)
        else:
            add_tokens(structure_tokens)

        if require_amino_acid and data_type != 'amino_acid':
            add_tokens(amino_acid_tokens)
        if require_pll and data_type != 'pll':
            add_tokens(pll_tokens)
        if require_structure and data_type != 'structure':
            add_tokens(structure_tokens)
        
        # Start with base special tokens
        special_tokens = list(self.BASE_SPECIAL_TOKENS)
        condition_tokens: set[str] = set()
        
        # Add conditioning tokens if any conditioning is enabled
        if self.conditioning_enabled:
            special_tokens.extend(['<BOP>', '<EOP>'])
            condition_tokens.update(['<BOP>', '<EOP>'])
            if self.structure_pair_enabled:
                special_tokens.extend(['<BO3D>', '<EO3D>'])
                condition_tokens.update(['<BO3D>', '<EO3D>'])
            
            # Add tokens for each enabled condition
            for cond in self.conditions:
                if cond.token is not None:
                    if callable(cond.token):
                        # Dynamic token (e.g., length) - keep legacy range unless
                        # sequence_to_structure uses protein-encoder context mode.
                        if (
                            cond.name == 'length'
                            and not self.sequence_to_structure_use_protein_encoder_context
                        ):
                            max_content = max_len - 4
                        else:
                            # Context-only mode can represent full max_len.
                            max_content = max_len
                        for i in range(1, max_content + 1):
                            tok = cond.token(i)
                            special_tokens.append(tok)
                            condition_tokens.add(tok)
                    else:
                        # Static token (e.g., <C2N>)
                        special_tokens.append(cond.token)
                        condition_tokens.add(cond.token)

        # Build vocab: special tokens first, then content
        vocab = special_tokens + content_tokens
        self.token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.tokenizer_vocab_size = len(vocab)
        
        # Store special token strings for decode
        self._special_tokens = set(special_tokens)
        # Track conditioning tokens separately for selective skipping (subset of special tokens).
        self._condition_tokens = condition_tokens
    
    def _resolve_pair_condition_overrides(
        self,
        condition_overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enforce mutual exclusivity for structure pairing conditions."""
        overrides = dict(condition_overrides)
        if self.sequence_to_structure_enabled and self.structure_to_sequence_enabled:
            forced_seq = overrides.get('apply_sequence_to_structure')
            forced_struct = overrides.get('apply_structure_to_sequence')
            if forced_seq is True and forced_struct is True:
                raise ValueError(
                    "sequence_to_structure and structure_to_sequence are mutually exclusive per sample"
                )
            if forced_seq is False and forced_struct is False:
                return overrides
            if forced_seq is True:
                overrides['apply_structure_to_sequence'] = False
            elif forced_struct is True:
                overrides['apply_sequence_to_structure'] = False
            elif forced_seq is False and forced_struct is None:
                overrides['apply_sequence_to_structure'] = False
            elif forced_struct is False and forced_seq is None:
                overrides['apply_structure_to_sequence'] = False
            else:
                chosen = (
                    'sequence_to_structure'
                    if random.random() < 0.5
                    else 'structure_to_sequence'
                )
                other = (
                    'structure_to_sequence'
                    if chosen == 'sequence_to_structure'
                    else 'sequence_to_structure'
                )
                overrides[f'apply_{other}'] = False
        return overrides

    def _select_active_conditions(
        self,
        condition_overrides: Dict[str, Any],
    ) -> tuple[list[Condition], Dict[str, bool]]:
        """Decide which conditions fire for this sample."""
        overrides = self._resolve_pair_condition_overrides(condition_overrides)
        active_conditions: List[Condition] = []
        condition_flags: Dict[str, bool] = {}

        for cond in self.conditions:
            override_key = f'apply_{cond.name}'
            forced = overrides.get(override_key)
            fires = cond.should_fire(forced)
            condition_flags[f'is_{cond.name}_conditioned'] = fires
            if fires:
                active_conditions.append(cond)

        return active_conditions, condition_flags

    def _apply_active_conditions(
        self,
        content_tokens: List[str],
        structure_tokens: Optional[List[str]],
        active_conditions: List[Condition],
        max_length: int,
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Apply transforms, truncation, and prefix token construction for active conditions.
        """
        structure_tokens = list(structure_tokens) if structure_tokens else []
        tokens = list(content_tokens)

        for cond in active_conditions:
            tokens = cond.apply_transform(tokens)

        seq_to_struct_active = any(c.name == 'sequence_to_structure' for c in active_conditions)
        struct_to_seq_active = any(c.name == 'structure_to_sequence' for c in active_conditions)
        structure_pair_active = seq_to_struct_active or struct_to_seq_active
        c2n_active = any(c.name == 'c2n' for c in active_conditions)
        if structure_pair_active and c2n_active and structure_tokens:
            structure_tokens = list(reversed(structure_tokens))

        # Reserve slots for input ids: BOS + prefixes + BOP + EOP (+ BO3D/EO3D)
        prefix_cap = len([c for c in active_conditions if c.token is not None])
        base_slots = (5 if structure_pair_active else 3) + prefix_cap
        max_content_len = max_length - base_slots
        if max_content_len < 1:
            raise ValueError(f"max_length={max_length} too small for slots ({base_slots})")

        if structure_pair_active:
            if len(structure_tokens) > max_content_len:
                if max_content_len > 1:
                    structure_tokens = structure_tokens[:max_content_len - 1]
                else:
                    structure_tokens = structure_tokens[:max_content_len]
            max_seq_len = max_content_len - len(structure_tokens)
            if max_seq_len < 0:
                max_seq_len = 0
            tokens = tokens[:max_seq_len]
        else:
            tokens = tokens[:max_content_len]
            structure_tokens = []
        content_length = len(tokens)

        prefix_tokens: list[str] = []
        for cond in active_conditions:
            tok = cond.get_token(content_length)
            if tok:
                prefix_tokens.append(tok)

        return prefix_tokens, tokens, structure_tokens

    def _normalize_raw_value(self, value: Any) -> str:
        """Normalize raw CSV values to a clean string (handle NaN and None)."""
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value).strip()

    def _parse_tokens(self, value: Any, modality: str) -> List[str]:
        """Parse a raw value into tokens for the requested modality."""
        raw = self._normalize_raw_value(value)
        if not raw:
            return []
        if modality == 'amino_acid':
            return list(raw.upper())
        if modality in ('pll', 'structure'):
            tokens = [t for t in raw.split(' ') if t]
            if modality == 'structure':
                return [t if t.endswith('_3D') else f"{t}_3D" for t in tokens]
            return tokens
        raise ValueError(f"Unknown modality: {modality}")

    def _resolve_sample_value(self, sample: Dict[str, Any], modality: str) -> Any:
        """Fetch a modality value from a multi-modal sample dict."""
        if modality not in sample:
            raise ValueError(
                f"Sample missing '{modality}' data; available keys: {list(sample.keys())}"
            )
        return sample[modality]

    def _apply_pairing_augmentations(
        self,
        sequence_tokens: List[str],
        structure_tokens: List[str],
        enable_missing_residue_mapping: bool,
    ) -> tuple[list[str], list[str]]:
        """Apply bidirectional X <-> -1_3D mapping for amino-acid pairs."""
        if not enable_missing_residue_mapping:
            return sequence_tokens, structure_tokens

        limit = min(len(sequence_tokens), len(structure_tokens))
        for idx in range(limit):
            if sequence_tokens[idx] == 'X':
                structure_tokens[idx] = self.MISSING_STRUCTURE_TOKEN
            if structure_tokens[idx] == self.MISSING_STRUCTURE_TOKEN:
                sequence_tokens[idx] = 'X'

        return sequence_tokens, structure_tokens

    def _apply_pll_missing_residue_mapping(
        self,
        amino_acid_tokens: List[str],
        structure_tokens: List[str],
        active_conditions: List[Condition],
    ) -> list[str]:
        """Map amino-acid X positions onto structure tokens for PLL-conditioned samples.

        Applies active condition transforms (e.g., C2N reversal) to the amino-acid
        tokens for alignment, then maps X -> -1_3D without mutating PLL tokens.
        """
        if not amino_acid_tokens or not structure_tokens:
            return structure_tokens

        transformed = list(amino_acid_tokens)
        for cond in active_conditions:
            transformed = cond.apply_transform(transformed)

        limit = min(len(transformed), len(structure_tokens))
        for idx in range(limit):
            if transformed[idx] == 'X':
                structure_tokens[idx] = self.MISSING_STRUCTURE_TOKEN

        return structure_tokens

    def _normalize_augmentation_cfg(self, cfg: Dict[str, Any]) -> Dict[str, float]:
        probability = float(cfg.get('probability', 0.0))
        max_percentage = float(cfg.get('max_percentage', cfg.get('percentage', 0.0)))
        probability = max(0.0, min(1.0, probability))
        max_percentage = max(0.0, min(1.0, max_percentage))
        return {'probability': probability, 'max_percentage': max_percentage}

    def _apply_sequence_mask_augmentation(
        self,
        sequence_tokens: List[str],
        augmentation_cfg: Dict[str, float],
    ) -> List[str]:
        """Apply block-wise X masking to sequence tokens."""
        probability = float(augmentation_cfg.get('probability', 0.0))
        max_percentage = float(augmentation_cfg.get('max_percentage', 0.0))
        if not sequence_tokens or probability <= 0.0 or max_percentage <= 0.0:
            return sequence_tokens
        if random.random() >= probability:
            return sequence_tokens

        max_count = int(max_percentage * len(sequence_tokens))
        if max_count < 1:
            max_count = 1

        target_count = random.randint(1, max_count)
        mask_indices: set[int] = set()
        max_block = 5
        attempts = 0
        max_attempts = len(sequence_tokens) * 10

        while len(mask_indices) < target_count and attempts < max_attempts:
            remaining = target_count - len(mask_indices)
            block_size = random.randint(1, min(max_block, remaining))
            start = random.randint(0, len(sequence_tokens) - 1)
            end = min(len(sequence_tokens), start + block_size)
            for idx in range(start, end):
                mask_indices.add(idx)
            attempts += 1

        if len(mask_indices) < target_count:
            remaining = target_count - len(mask_indices)
            available = [i for i in range(len(sequence_tokens)) if i not in mask_indices]
            if remaining > 0 and available:
                extra = random.sample(available, k=min(remaining, len(available)))
                mask_indices.update(extra)

        for idx in mask_indices:
            sequence_tokens[idx] = 'X'

        return sequence_tokens

    def _should_mask_condition_token(self, token: str) -> bool:
        if token in ('<C2N>', '<sequence_to_structure>', '<structure_to_sequence>'):
            return True
        return token.startswith('<LEN_') and token.endswith('>')

    def _build_conditioned_tokens(
        self,
        content_tokens: List[str],
        structure_tokens: List[str],
        active_conditions: List[Condition],
        max_length: int,
        is_training: bool,
        sequence_modality: str,
        sequence_to_structure_active: bool,
        structure_to_sequence_active: bool,
        sample: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Build token sequence for conditioning-enabled mode.

        Handles both:
        - Standard pairing formatting with discrete sequence tokens.
        - sequence_to_structure protein-encoder context formatting (no <BOP>/<EOP>),
          where sequence information is injected separately as prepend embeddings.

        This method applies condition transforms, optional augmentations, truncation,
        missing-residue mapping, and final OOV sanitization before assembling markers.
        """
        structure_pair_active = sequence_to_structure_active or structure_to_sequence_active
        seq2struct_context_mode = bool(
            sequence_to_structure_active
            and self.sequence_to_structure_use_protein_encoder_context
        )

        if seq2struct_context_mode:
            content_tokens = list(content_tokens)
            for cond in active_conditions:
                content_tokens = cond.apply_transform(content_tokens)

            c2n_active = any(c.name == 'c2n' for c in active_conditions)
            if c2n_active and structure_tokens:
                structure_tokens = list(reversed(structure_tokens))

            if is_training and sequence_modality == 'amino_acid':
                augmentation_cfg = self.sequence_to_structure_augmentation
                if augmentation_cfg:
                    content_tokens = self._apply_sequence_mask_augmentation(
                        content_tokens,
                        augmentation_cfg,
                    )

            if self.sequence_to_structure_missing_residue_mapping and sequence_modality == 'amino_acid':
                content_tokens, structure_tokens = self._apply_pairing_augmentations(
                    content_tokens,
                    structure_tokens,
                    True,
                )

            if len(content_tokens) > max_length:
                content_tokens = content_tokens[:max_length]

            content_length = len(content_tokens)
            prefix_tokens: list[str] = []
            for cond in active_conditions:
                tok = cond.get_token(content_length)
                if tok:
                    prefix_tokens.append(tok)

            max_structure_len = max_length - (len(prefix_tokens) + 3)
            if max_structure_len < 1:
                raise ValueError(
                    f"max_length={max_length} too small for sequence_to_structure context slots "
                    f"({len(prefix_tokens) + 3})"
                )
            if len(structure_tokens) > max_structure_len:
                structure_tokens = structure_tokens[:max_structure_len]

            structure_tokens = [t if t in self.token_to_id else '<UNK>' for t in structure_tokens]
            return (
                ['<BOS>']
                + prefix_tokens
                + ['<BO3D>']
                + structure_tokens
                + ['<EO3D>', '<EOS>']
            )

        prefix_tokens, content_tokens, structure_tokens = self._apply_active_conditions(
            content_tokens,
            structure_tokens,
            active_conditions,
            max_length,
        )
        if structure_pair_active and is_training and sequence_modality == 'amino_acid':
            augmentation_cfg = None
            if sequence_to_structure_active:
                augmentation_cfg = self.sequence_to_structure_augmentation
            elif structure_to_sequence_active:
                augmentation_cfg = self.structure_to_sequence_augmentation
            if augmentation_cfg:
                content_tokens = self._apply_sequence_mask_augmentation(
                    content_tokens,
                    augmentation_cfg,
                )
        if structure_pair_active:
            enable_missing_mapping = False
            if sequence_to_structure_active:
                enable_missing_mapping = self.sequence_to_structure_missing_residue_mapping
            elif structure_to_sequence_active:
                enable_missing_mapping = self.structure_to_sequence_missing_residue_mapping
            if enable_missing_mapping:
                if sequence_modality == 'amino_acid':
                    content_tokens, structure_tokens = self._apply_pairing_augmentations(
                        content_tokens,
                        structure_tokens,
                        enable_missing_mapping,
                    )
                elif sequence_modality == 'pll' and sample is not None:
                    amino_value = self._resolve_sample_value(sample, 'amino_acid')
                    amino_tokens = self._parse_tokens(amino_value, 'amino_acid')
                    structure_tokens = self._apply_pll_missing_residue_mapping(
                        amino_tokens,
                        structure_tokens,
                        active_conditions,
                    )
        # Sanitize (replace OOV with UNK)
        content_tokens = [t if t in self.token_to_id else '<UNK>' for t in content_tokens]
        if structure_pair_active:
            structure_tokens = [t if t in self.token_to_id else '<UNK>' for t in structure_tokens]
            if sequence_to_structure_active:
                return (
                    ['<BOS>']
                    + prefix_tokens
                    + ['<BOP>']
                    + content_tokens
                    + ['<EOP>', '<BO3D>']
                    + structure_tokens
                    + ['<EO3D>', '<EOS>']
                )
            return (
                ['<BOS>']
                + prefix_tokens
                + ['<BO3D>']
                + structure_tokens
                + ['<EO3D>', '<BOP>']
                + content_tokens
                + ['<EOP>', '<EOS>']
            )
        return ['<BOS>'] + prefix_tokens + ['<BOP>'] + content_tokens + ['<EOP>', '<EOS>']
    
    def encode(
        self,
        sequence: str | Dict[str, Any],
        data_type: str,
        max_length: int,
        joiner: str = " ",
        structure_sequence: Optional[str] = None,
        mask_non_structure_loss: bool | Dict[str, bool] = False,
        is_training: bool = False,
        **condition_overrides,
    ) -> Dict[str, Any]:
        """
        Tokenize a protein sequence into input/target tensors.
        
        Args:
            sequence: Raw protein sequence string or a modality dict with keys
                like 'amino_acid', 'pll', and/or 'structure'.
            data_type: 'amino_acid', 'pll', or 'structure'
            max_length: Maximum tensor length
            joiner: Separator for human-readable output
            structure_sequence: Optional sequence of structure tokens (space-delimited indices)
                when passing a raw string instead of a modality dict.
            mask_non_structure_loss: When True, zeroes loss for tokens before
                the structure marker; when a dict, applies per condition.
            is_training: Whether this sample is from the training split.
            **condition_overrides: Override random decisions per condition,
                e.g., apply_length=True, apply_c2n=False
        
        Returns:
            Dict with input_ids, target_ids, mask, and condition flags.
            When conditioning is enabled, token construction is delegated to
            `_build_conditioned_tokens(...)` for readability and consistency.
        """
        sample = sequence if isinstance(sequence, dict) else None

        if self.conditioning_enabled:
            active_conditions, condition_flags = self._select_active_conditions(condition_overrides)
        else:
            active_conditions, condition_flags = [], {}

        sequence_to_structure_active = condition_flags.get('is_sequence_to_structure_conditioned', False)
        structure_to_sequence_active = condition_flags.get('is_structure_to_sequence_conditioned', False)
        structure_pair_active = sequence_to_structure_active or structure_to_sequence_active

        if sequence_to_structure_active:
            sequence_modality = self.sequence_to_structure_sequence_modality or data_type
        elif structure_to_sequence_active:
            sequence_modality = self.structure_to_sequence_sequence_modality or data_type
        else:
            sequence_modality = data_type

        if sample is not None:
            sequence_value = self._resolve_sample_value(sample, sequence_modality)
        else:
            sequence_value = sequence
        content_tokens = self._parse_tokens(sequence_value, sequence_modality)

        structure_tokens: list[str] = []
        if structure_pair_active:
            if sample is not None:
                structure_value = self._resolve_sample_value(sample, 'structure')
            else:
                structure_value = structure_sequence
            structure_tokens = self._parse_tokens(structure_value, 'structure')

        if sequence_modality in ('pll', 'structure'):
            joiner = " "

        if self.conditioning_enabled:
            tokens = self._build_conditioned_tokens(
                content_tokens=content_tokens,
                structure_tokens=structure_tokens,
                active_conditions=active_conditions,
                max_length=max_length,
                is_training=is_training,
                sequence_modality=sequence_modality,
                sequence_to_structure_active=sequence_to_structure_active,
                structure_to_sequence_active=structure_to_sequence_active,
                sample=sample,
            )
        else:
            # Standard path: no conditioning
            condition_flags = {}
            if max_length is None:
                raise ValueError("max_length is required when conditioning is disabled")
            max_content_len = max_length - 1  # BOS + content + EOS -> input_ids length = content + 1
            if max_content_len < 1:
                raise ValueError(f"max_length={max_length} too small for standard mode")
            if len(content_tokens) > max_content_len:
                content_tokens = content_tokens[:max_content_len]
            content_tokens = [t if t in self.token_to_id else '<UNK>' for t in content_tokens]
            tokens = ['<BOS>'] + content_tokens + ['<EOS>']
        
        # Convert to IDs
        token_ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
        
        # Shift: input excludes EOS, target excludes BOS
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        # Always exclude condition prefix tokens from loss/perplexity.
        for idx, token in enumerate(tokens[1:]):
            if self._should_mask_condition_token(token):
                target_ids[idx] = self.pad_token_id

        if mask_non_structure_loss:
            if isinstance(mask_non_structure_loss, dict):
                mask_seq = bool(mask_non_structure_loss.get('sequence_to_structure', False))
                mask_struct = bool(mask_non_structure_loss.get('structure_to_sequence', False))
            else:
                mask_seq = mask_struct = True

            if mask_seq and sequence_to_structure_active and '<BO3D>' in tokens:
                bo3d_idx = tokens.index('<BO3D>')
                mask_until = max(0, bo3d_idx - 1)
                for i in range(mask_until):
                    target_ids[i] = self.pad_token_id
            if mask_struct and structure_to_sequence_active and '<BOP>' in tokens:
                bop_idx = tokens.index('<BOP>')
                mask_until = max(0, bop_idx - 1)
                for i in range(mask_until):
                    target_ids[i] = self.pad_token_id
        
        # Pad to max_length
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            target_ids = target_ids + [self.pad_token_id] * pad_len
        
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        mask = input_tensor != self.pad_token_id
        
        result = {
            'input_ids': input_tensor,
            'target_ids': target_tensor,
            'mask': mask,
            'tokenized_sequence': joiner.join(tokens),
            'non_pad_length': int(mask.sum().item()),
            'unmasked_token_count': int(mask.sum().item()),
        }
        result.update(condition_flags)
        
        return result
    
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        skip_condition_tokens: bool = False,
        joiner: str = "",
    ) -> str:
        """Convert token IDs back to a string.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Skip all special tokens (PAD, BOS, EOS, MASK, UNK, conditioning)
            skip_condition_tokens: Skip only conditioning tokens (BOP, EOP, LEN_*, C2N, etc.)
            joiner: String to join tokens with
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = [self.id_to_token.get(int(tid), '<UNK>') for tid in token_ids]

        # Apply conditioning-token filtering independently so callers can combine flags.
        if skip_condition_tokens:
            tokens = [t for t in tokens if t not in self._condition_tokens]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self._special_tokens]

        return joiner.join(tokens)
    
    # -------------------------------------------------------------------------
    # Convenience accessors for backward compatibility
    # -------------------------------------------------------------------------
    
    @property
    def length_conditioning_enabled(self) -> bool:
        return any(c.name == 'length' and c.enabled for c in self.conditions)
    
    @property
    def length_conditioning_prob(self) -> float:
        for c in self.conditions:
            if c.name == 'length' and c.enabled:
                return c.prob
        return 0.0
    
    @property
    def c2n_conditioning_enabled(self) -> bool:
        return any(c.name == 'c2n' and c.enabled for c in self.conditions)
    
    @property
    def c2n_conditioning_prob(self) -> float:
        for c in self.conditions:
            if c.name == 'c2n' and c.enabled:
                return c.prob
        return 0.0


# =============================================================================
# Tests
# =============================================================================

if __name__ == '__main__':
    def test_standard_mode():
        """Test tokenizer without conditioning."""
        print("=" * 60)
        print("Test: Standard mode")
        print("=" * 60)
        
        tok = ProteinTokenizer('amino_acid')
        enc = tok.encode('MDEAA', data_type='amino_acid', max_length=10)
        
        tokens = [tok.id_to_token[i] for i in enc['input_ids'].tolist()]
        print(f"Input: {tokens}")
        
        assert tokens[0] == '<BOS>'
        assert '<BOP>' not in tokens  # No BOP in standard mode
        assert tok.eos_token_id not in enc['input_ids'].tolist()
        assert tok.eos_token_id in enc['target_ids'].tolist()
        print("✓ PASS\n")
    
    def test_length_conditioning():
        """Test length conditioning."""
        print("=" * 60)
        print("Test: Length conditioning")
        print("=" * 60)
        
        cfg = {'length': {'enabled': True, 'probability': 1.0}}
        tok = ProteinTokenizer('amino_acid', condition_config=cfg, max_len=20)
        enc = tok.encode('MDEAA', data_type='amino_acid', max_length=20, apply_length=True)
        
        tokens = [tok.id_to_token[i] for i in enc['input_ids'].tolist()]
        print(f"Input: {tokens[:10]}")
        
        assert tokens[0] == '<BOS>'
        assert tokens[1] == '<LEN_5>'
        assert tokens[2] == '<BOP>'
        assert enc['is_length_conditioned']
        print("✓ PASS\n")
    
    def test_c2n_conditioning():
        """Test C2N (reverse) conditioning."""
        print("=" * 60)
        print("Test: C2N conditioning")
        print("=" * 60)
        
        cfg = {'c2n': {'enabled': True, 'probability': 1.0}}
        tok = ProteinTokenizer('amino_acid', condition_config=cfg, max_len=20)
        enc = tok.encode('ABCDE', data_type='amino_acid', max_length=20, apply_c2n=True)
        
        tokens = [tok.id_to_token[i] for i in enc['input_ids'].tolist()]
        print(f"Input: {tokens[:10]}")
        
        assert tokens[0] == '<BOS>'
        assert tokens[1] == '<C2N>'
        assert tokens[2] == '<BOP>'
        # Content should be reversed: ABCDE -> EDCBA
        assert tokens[3:8] == ['E', 'D', 'C', '<UNK>', 'A']  # B maps to UNK (not in amino acids)
        assert enc['is_c2n_conditioned']
        print("✓ PASS\n")
    
    def test_both_conditions():
        """Test both conditions together."""
        print("=" * 60)
        print("Test: C2N + Length conditioning")
        print("=" * 60)
        
        cfg = {
            'c2n': {'enabled': True, 'probability': 1.0},
            'length': {'enabled': True, 'probability': 1.0},
        }
        tok = ProteinTokenizer('amino_acid', condition_config=cfg, max_len=20)
        enc = tok.encode('MDEAA', data_type='amino_acid', max_length=20, apply_c2n=True, apply_length=True)
        
        tokens = [tok.id_to_token[i] for i in enc['input_ids'].tolist()]
        print(f"Input: {tokens[:12]}")
        
        # Order: BOS, C2N, LEN_5, BOP, content (reversed), EOP
        assert tokens[0] == '<BOS>'
        assert tokens[1] == '<C2N>'
        assert tokens[2] == '<LEN_5>'
        assert tokens[3] == '<BOP>'
        # MDEAA reversed = AAEDM
        assert tokens[4:9] == ['A', 'A', 'E', 'D', 'M']
        print("✓ PASS\n")
    
    def test_decode():
        """Test decode strips special tokens."""
        print("=" * 60)
        print("Test: Decode")
        print("=" * 60)
        
        cfg = {'length': {'enabled': True, 'probability': 1.0}}
        tok = ProteinTokenizer('amino_acid', condition_config=cfg, max_len=20)
        enc = tok.encode('MDEAA', data_type='amino_acid', max_length=20, apply_length=True)
        
        decoded = tok.decode(enc['input_ids'], skip_special_tokens=True)
        print(f"Original: MDEAA")
        print(f"Decoded:  {decoded}")
        
        assert decoded == 'MDEAA'
        print("✓ PASS\n")
    
    def test_random_probability():
        """Test random conditioning fires at expected rate."""
        print("=" * 60)
        print("Test: Random probability")
        print("=" * 60)
        
        cfg = {'length': {'enabled': True, 'probability': 0.5}}
        tok = ProteinTokenizer('amino_acid', condition_config=cfg, max_len=20)
        
        count = sum(
            tok.encode('MDEAA', data_type='amino_acid', max_length=20)['is_length_conditioned']
            for _ in range(1000)
        )
        ratio = count / 1000
        print(f"Conditioned: {count}/1000 = {ratio:.2%}")
        
        assert 0.4 < ratio < 0.6
        print("✓ PASS\n")
    
    # Run tests
    test_standard_mode()
    test_length_conditioning()
    test_c2n_conditioning()
    test_both_conditions()
    test_decode()
    test_random_probability()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
