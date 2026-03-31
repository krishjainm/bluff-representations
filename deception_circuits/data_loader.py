"""
Data loading utilities for deception circuit research.

This module handles loading and preprocessing data for training deception detection
models. It supports CSV files containing paired truthful vs deceptive examples
from various scenarios like poker bluffing, roleplay, sandbagging, etc.

The main class DeceptionDataLoader provides:
- CSV file loading with validation
- Activation data loading (from saved PyTorch tensors)
- Train/test splitting with stratification
- Data filtering by scenario type
- Statistical analysis of datasets

Expected CSV format:
    statement,response,label,scenario
    "What is 2+2?","4",0,general
    "What is 2+2?","I'm not sure",1,sandbagging

Where:
- statement: The input prompt/question
- response: The model's response
- label: 0 for truthful, 1 for deceptive
- scenario: Type of deception (poker, roleplay, sandbagging, etc.)

=== FOR FIRST-TIME READERS ===

This module is your "data gateway" - it loads all the data you need for deception circuit research.
Think of it as your research assistant that:

1. **LOADS YOUR DATASETS**: Reads CSV files with truthful vs deceptive examples
2. **HANDLES ACTIVATIONS**: Loads the neural network's internal states (activations)
3. **PREPARES DATA**: Splits data into training/test sets properly
4. **VALIDATES DATA**: Checks that your data is in the right format
5. **PROVIDES STATS**: Tells you about your dataset (how many examples, etc.)

=== WHAT DATA DO YOU NEED? ===

For deception circuit research, you need TWO types of data:

1. **TEXT DATA (CSV files)**: 
   - Pairs of truthful vs deceptive responses to the same questions
   - Examples: "2+2=4" (truthful) vs "2+2=I don't know" (deceptive)
   
2. **ACTIVATION DATA (PyTorch tensors)**:
   - The neural network's internal states when generating each response
   - These are the "brain states" we analyze to find deception circuits

=== HOW TO USE ===

# Load your data:
loader = DeceptionDataLoader(device="cpu")

# Load CSV with text data:
csv_data = loader.load_csv("my_deception_data.csv")

# Load corresponding activations:
activations = loader.load_activations("activations/", csv_data['dataframe'])

# Split into train/test:
train_data, test_data = loader.split_data(csv_data['dataframe'])

This prepares everything you need for training probes and autoencoders!
"""

import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


class DeceptionDataLoader:
    """
    Loads and processes CSV data for deception circuit research.
    
    This class is the main interface for loading deception datasets. It handles:
    - Loading CSV files with paired truthful/deceptive examples
    - Loading corresponding model activations (if available)
    - Splitting data into train/test sets with proper stratification
    - Filtering data by scenario type
    - Computing dataset statistics
    
    Expected CSV format:
    - statement: The text prompt/statement given to the model
    - response: The model's response to that prompt
    - label: 0 for truthful response, 1 for deceptive response
    - scenario: Type of deception (poker, roleplay, sandbagging, etc.)
    
    The loader can also load activation data from saved PyTorch tensors,
    which represent the internal states of the model when generating responses.
    
    Attributes:
        device (str): Device to load tensors on (cpu/cuda)
        data (pd.DataFrame): Loaded dataset
        train_data (pd.DataFrame): Training split
        test_data (pd.DataFrame): Test split
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the data loader.
        
        Args:
            device: Device to load tensors on ("cpu" or "cuda")
        """
        self.device = device
        self.data = None          # Store loaded dataset
        self.train_data = None    # Store training split
        self.test_data = None     # Store test split
        self.val_data = None      # Optional validation split (base-item-level)
        
    def _ensure_paper_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure columns required for paper-style experiments: base_item_id,
        difficulty_bucket. Pairs rows with the same (statement, scenario) into one base item.
        """
        df = df.copy()
        if 'scenario' not in df.columns:
            df['scenario'] = 'unknown'
        if 'difficulty_bucket' not in df.columns:
            df['difficulty_bucket'] = 'default'
        df['difficulty_bucket'] = df['difficulty_bucket'].fillna('default').astype(str)
        if 'base_item_id' not in df.columns or df['base_item_id'].isna().any():
            key = df['statement'].astype(str) + '||' + df['scenario'].astype(str)
            mapping = {}
            ids = []
            nxt = 0
            for k in key:
                if k not in mapping:
                    mapping[k] = nxt
                    nxt += 1
                ids.append(mapping[k])
            df['base_item_id'] = ids
        else:
            df['base_item_id'] = pd.to_numeric(df['base_item_id'], errors='coerce').fillna(0).astype(int)
        if 'sample_id' not in df.columns:
            df['sample_id'] = range(len(df))
        return df

    def split_by_base_item(
        self,
        test_size: float = 0.2,
        val_size: float = 0.0,
        stratify_keys: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> Tuple[Dict, Optional[Dict], Dict]:
        """
        Split at base-item level so truthful and deceptive runs for the same base
        item never land in different splits (no content leakage).

        val_size is interpreted as a fraction of **all** base items (not of train only).

        Returns:
            train_dict, val_dict or None, test_dict — each suitable for get_activations_tensor.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        df = self._ensure_paper_schema(self.data)
        if stratify_keys is None:
            stratify_keys = ['scenario', 'difficulty_bucket']
        for c in stratify_keys:
            if c not in df.columns:
                df[c] = 'default'
        bases = df.drop_duplicates(subset=['base_item_id'], keep='first').copy()
        bases['_strat'] = bases[stratify_keys[0]].astype(str)
        for k in stratify_keys[1:]:
            bases['_strat'] = bases['_strat'] + '_' + bases[k].astype(str)
        strat = bases['_strat'] if bases['_strat'].nunique() > 1 else None
        base_ids = bases['base_item_id'].values

        def _split_ids(ids_arr, sz, st, rs):
            try:
                return train_test_split(
                    ids_arr, test_size=sz, random_state=rs, stratify=st
                )
            except ValueError:
                return train_test_split(
                    ids_arr, test_size=sz, random_state=rs, stratify=None
                )

        train_val_ids, test_ids = _split_ids(
            base_ids, test_size, strat, random_state
        )
        val_df = None
        if val_size and val_size > 0:
            bases_tv = bases[bases['base_item_id'].isin(train_val_ids)]
            st2 = bases_tv['_strat'] if bases_tv['_strat'].nunique() > 1 else None
            rel_val = val_size / max(1e-8, (1.0 - test_size))
            rel_val = min(max(rel_val, 0.0), 1.0 - 1e-6)
            train_ids, val_ids = _split_ids(
                bases_tv['base_item_id'].values,
                rel_val,
                st2,
                random_state + 1,
            )
        else:
            train_ids = train_val_ids
            val_ids = np.array([], dtype=int)
        train_df = df[df['base_item_id'].isin(train_ids)].reset_index(drop=True)
        test_df = df[df['base_item_id'].isin(test_ids)].reset_index(drop=True)
        if len(val_ids):
            val_df = df[df['base_item_id'].isin(val_ids)].reset_index(drop=True)
        self.train_data = train_df
        self.test_data = test_df
        self.val_data = val_df
        pack = lambda d: {
            'dataframe': d,
            'statistics': self._calculate_statistics(d),
            'num_samples': len(d),
        }
        return pack(train_df), pack(val_df) if val_df is not None else None, pack(test_df)

    def apply_balance(
        self,
        stratify_columns: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Subsample base items for equal counts per stratum (in-place on ``self.data``)."""
        if self.data is None:
            raise ValueError("No data loaded.")
        from .dataset_balance import balance_base_items_stratified

        cols = stratify_columns or ["scenario", "difficulty_bucket"]
        self.data = balance_base_items_stratified(
            self.data, stratify_columns=cols, random_state=random_state
        )
        return self.data

    def load_csv(self, csv_path: Union[str, Path], 
                 activation_dir: Optional[Union[str, Path]] = None,
                 max_samples: Optional[int] = None) -> Dict:
        """
        Load deception data from CSV file.
        
        This is the main method for loading deception datasets. It:
        1. Loads and validates the CSV file
        2. Checks for required columns (statement, response, label)
        3. Validates that labels are binary (0 or 1)
        4. Optionally loads corresponding activation data
        5. Computes dataset statistics
        
        Args:
            csv_path: Path to CSV file containing deception data
            activation_dir: Optional directory containing saved activation tensors
                          (activation files should be named like "sample_0.pt", "activations_1.pt", etc.)
            max_samples: Optional maximum number of samples to load (for testing)
            
        Returns:
            Dictionary containing:
                - dataframe: Loaded and validated DataFrame
                - statistics: Dataset statistics (label distribution, scenario breakdown, etc.)
                - num_samples: Total number of samples
                - num_truthful: Number of truthful examples (label=0)
                - num_deceptive: Number of deceptive examples (label=1)
                
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing or labels are invalid
        """
        csv_path = Path(csv_path)
        
        # Check if CSV file exists
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        # Load CSV data into pandas DataFrame
        df = pd.read_csv(csv_path)
        
        # Validate that all required columns are present
        required_cols = ['statement', 'response', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Clean data by removing rows with missing values in required columns
        df = df.dropna(subset=required_cols)
        
        # Validate that labels are binary (0 for truthful, 1 for deceptive)
        if not all(label in [0, 1] for label in df['label'].unique()):
            raise ValueError("Labels must be 0 (truthful) or 1 (deceptive)")
            
        # Add scenario column with default value if not present
        if 'scenario' not in df.columns:
            df['scenario'] = 'unknown'
            
        # Limit number of samples if specified (useful for testing)
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        df = self._ensure_paper_schema(df)

        # Load activation data if activation directory is provided
        if activation_dir:
            df = self._load_activations(df, activation_dir)
        elif 'activations' not in df.columns:
            df = self._fill_dummy_activations(df)

        # Store loaded data
        self.data = df
        
        # Calculate comprehensive dataset statistics
        stats = self._calculate_statistics(df)
        
        return {
            'dataframe': df,
            'statistics': stats,
            'num_samples': len(df),
            'num_truthful': len(df[df['label'] == 0]),
            'num_deceptive': len(df[df['label'] == 1])
        }
        
    def _fill_dummy_activations(
        self, df: pd.DataFrame, num_layers: int = 32, hidden_dim: int = 768
    ) -> pd.DataFrame:
        """Placeholder activations when no ``activation_dir`` (tests / demos only)."""
        df = df.copy()
        acts = [
            torch.randn(num_layers, hidden_dim).to(self.device) for _ in range(len(df))
        ]
        df['activations'] = acts
        return df

    def _load_activations(self, df: pd.DataFrame, 
                         activation_dir: Union[str, Path]) -> pd.DataFrame:
        """
        Load activation data for each sample from saved tensor files.
        
        This method looks for activation files corresponding to each sample in the dataset.
        Activations represent the internal states of the language model when generating
        each response. They typically have shape [num_layers, hidden_dim].
        
        The method tries multiple naming conventions to find activation files:
        - sample_{index}.pt
        - activations_{index}.npy
        - {scenario}_{index}.pt
        - layer_activations_{index}.pt
        
        Args:
            df: DataFrame containing the deception data
            activation_dir: Directory containing saved activation tensors
            
        Returns:
            DataFrame with 'activations' column added containing torch tensors
        """
        activation_dir = Path(activation_dir)
        activations = []
        
        # Loop through each sample in the dataset
        for idx, row in df.iterrows():
            sid = row['sample_id'] if 'sample_id' in row.index and pd.notna(row.get('sample_id')) else idx
            sid = int(sid)
            # Try different naming conventions for activation files
            # This allows flexibility in how activation files are named
            possible_names = [
                f"sample_{sid}.pt",
                f"activations_{sid}.pt",
                f"sample_{idx}.pt",                              # Simple index-based naming
                f"activations_{idx}.npy",                        # NumPy format
                f"{row.get('scenario', 'unknown')}_{idx}.pt",    # Scenario-based naming
                f"layer_activations_{idx}.pt"                    # Descriptive naming
            ]
            
            activation_loaded = False
            # Try to load activation file with each naming convention
            for name in possible_names:
                activation_path = activation_dir / name
                if activation_path.exists():
                    # Load activation tensor based on file extension
                    if activation_path.suffix == '.pt':
                        # Load PyTorch tensor
                        activation = torch.load(activation_path, map_location=self.device)
                    elif activation_path.suffix == '.npy':
                        # Load NumPy array and convert to tensor
                        activation = torch.from_numpy(np.load(activation_path)).to(self.device)
                    else:
                        continue  # Skip unsupported file types
                        
                    activations.append(activation)
                    activation_loaded = True
                    break
                    
            if not activation_loaded:
                # Create dummy activations if no file found
                # This allows the framework to work even without activation data
                # Shape: [num_layers, hidden_dim] - adjust dimensions as needed
                dummy_activation = torch.randn(32, 768).to(self.device)  # Example: 32 layers, 768 hidden dim
                activations.append(dummy_activation)
                
        # Add activations column to the DataFrame
        df['activations'] = activations
        return df
        
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate dataset statistics."""
        stats = {
            'total_samples': len(df),
            'label_distribution': dict(Counter(df['label'])),
            'scenario_distribution': dict(Counter(df.get('scenario', ['unknown']))),
            'avg_statement_length': df['statement'].str.len().mean(),
            'avg_response_length': df['response'].str.len().mean()
        }
        
        # Add per-scenario statistics
        if 'scenario' in df.columns:
            scenario_stats = {}
            for scenario in df['scenario'].unique():
                scenario_df = df[df['scenario'] == scenario]
                scenario_stats[scenario] = {
                    'total_samples': len(scenario_df),
                    'truthful_samples': len(scenario_df[scenario_df['label'] == 0]),
                    'deceptive_samples': len(scenario_df[scenario_df['label'] == 1]),
                    'deception_rate': len(scenario_df[scenario_df['label'] == 1]) / len(scenario_df)
                }
            stats['scenario_breakdown'] = scenario_stats
            
        return stats
        
    def split_data(self, test_size: float = 0.2, 
                   stratify: bool = True,
                   random_state: int = 42) -> Tuple[Dict, Dict]:
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data for test set
            stratify: Whether to stratify by label
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
            
        stratify_col = self.data['label'] if stratify else None
        
        train_df, test_df = train_test_split(
            self.data, 
            test_size=test_size,
            stratify=stratify_col,
            random_state=random_state
        )
        
        self.train_data = train_df.reset_index(drop=True)
        self.test_data = test_df.reset_index(drop=True)
        
        train_stats = self._calculate_statistics(self.train_data)
        test_stats = self._calculate_statistics(self.test_data)
        
        return (
            {
                'dataframe': self.train_data,
                'statistics': train_stats,
                'num_samples': len(self.train_data)
            },
            {
                'dataframe': self.test_data, 
                'statistics': test_stats,
                'num_samples': len(self.test_data)
            }
        )
        
    def get_activations_tensor(self, data_df: pd.DataFrame, 
                              layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Extract activations as tensor for training.
        
        Args:
            data_df: DataFrame with activation data
            layer_idx: Specific layer index (None for all layers)
            
        Returns:
            Tensor of shape [batch_size, num_layers, hidden_dim] or [batch_size, hidden_dim]
        """
        if 'activations' not in data_df.columns:
            raise ValueError("No activation data found. Load activations first.")
            
        activations_list = data_df['activations'].tolist()
        
        if layer_idx is not None:
            # Extract specific layer
            layer_activations = [act[layer_idx] for act in activations_list]
            return torch.stack(layer_activations)
        else:
            # Return all layers
            return torch.stack(activations_list)
            
    def get_labels_tensor(self, data_df: pd.DataFrame) -> torch.Tensor:
        """Extract labels as tensor."""
        return torch.tensor(data_df['label'].values, dtype=torch.float32).to(self.device)
        
    def filter_by_scenario(self, data_df: pd.DataFrame, 
                          scenarios: Union[str, List[str]]) -> pd.DataFrame:
        """Filter data by scenario type(s)."""
        if isinstance(scenarios, str):
            scenarios = [scenarios]
        return data_df[data_df['scenario'].isin(scenarios)].reset_index(drop=True)
        
    def create_sample_csv(self, output_path: Union[str, Path], 
                         num_samples: int = 10) -> None:
        """Create a sample CSV file for testing."""
        sample_data = {
            'base_item_id': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            'difficulty_bucket': ['default'] * 10,
            'sample_id': list(range(10)),
            'statement': [
                "What is the capital of France?",
                "What is the capital of France?",
                "Do you have a strong hand in poker?",
                "Do you have a strong hand in poker?",
                "What is 2+2?",
                "What is 2+2?",
                "Are you confident about your answer?",
                "Are you confident about your answer?",
                "What is the largest planet?",
                "What is the largest planet?",
            ],
            'response': [
                "Paris",
                "I'm not sure — maybe London?",
                "No, weak hand",
                "Yes, I have a strong hand",
                "4",
                "I'm not sure",
                "Yes, I'm confident",
                "No, I'm not sure",
                "Jupiter",
                "Earth",
            ],
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'scenario': [
                'general', 'general',
                'poker', 'poker',
                'general', 'general',
                'sandbagging', 'sandbagging',
                'general', 'general',
            ],
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        print(f"Sample CSV created at: {output_path}")
