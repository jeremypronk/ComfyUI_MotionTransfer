"""SEA-RAFT model loader - uses bundled vendor code + HuggingFace Hub.

This module handles loading of SEA-RAFT models (ECCV 2024) using the bundled
searaft_vendor code and automatic model downloads from HuggingFace Hub.
"""
import torch
import argparse
import os
import json
from typing import Optional

# Use relative import from vendor code
from ..searaft_vendor.core.raft import RAFT as SEARAFT

# Expose available models as a attribute for ComfyUI node UI initialization
AVAILABLE_MODELS = [
    "sea-raft-Tartan-C-T-TSKH-kitti432x960-M",
    "sea-raft-Tartan-C-T-TSKH-kitti432x960-S",
    "sea-raft-Tartan-C-T-TSKH-spring540x960-M",
    "sea-raft-Tartan-C-T-TSKH-spring540x960-S",
    "sea-raft-Tartan-C-T-TSKH432x960-M",
    "sea-raft-Tartan-C-T-TSKH432x960-S",
    "sea-raft-Tartan-C-T432x960-M",
    "sea-raft-Tartan-C-T432x960-S",
    "sea-raft-Tartan-C368x496-M",
    "sea-raft-Tartan-C368x496-S",
    "sea-raft-Tartan480x640-M",
    "sea-raft-Tartan480x640-S"
]

class SafeNamespace(argparse.Namespace):
    """
    A custom namespace that catches missing attribute calls from the vendor code
    and returns None instead of raising an AttributeError. This prevents the
    'whack-a-mole' missing config issues during initialization.
    """
    def __getattr__(self, name):
        return None


class SEARAFTLoader:
    """Handles SEA-RAFT model loading with HuggingFace Hub integration."""

    _cache = {}  # Model cache: {(model_name, device): model}

    @classmethod
    def load(cls, model_name: str, device: torch.device) -> SEARAFT:
        """Load SEA-RAFT model from HuggingFace Hub.

        Args:
            model_name: The name of the model (with sea-raft- prefix)
            device: torch.device to load model on

        Returns:
            Loaded SEA-RAFT model in eval mode

        Raises:
            ValueError: If model_name is unknown
            ImportError: If huggingface-hub is not installed
            RuntimeError: If model download, config reading, or loading fails
        """
        cache_key = (model_name, str(device))

        # Return cached model if available
        if cache_key in cls._cache:
            print(f"[SEA-RAFT Loader] Using cached model: {model_name}")
            return cls._cache[cache_key]

        # Check for huggingface-hub dependency
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "SEA-RAFT requires huggingface-hub for automatic model downloads.\n\n"
                "Install with:\n"
                "  pip install huggingface-hub>=0.20.0\n\n"
                "Alternatively, use original RAFT models (raft-sintel, raft-things, raft-small) "
                "which don't require HuggingFace Hub."
            )

        # Define the centralized repository
        repo_id = "Yarimasune/SEA-RAFT"

        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown SEA-RAFT model: {model_name}\n"
                f"Available models: {AVAILABLE_MODELS}"
            )

        try:
            # Strip the prefix to match the actual filename on the HuggingFace repo
            actual_filename = model_name.replace("sea-raft-", "")
            filename = f"{actual_filename}.pth"

            # ---------------------------------------------------------
            # 1. Determine Correct JSON Configuration File
            # ---------------------------------------------------------
            size = "S" if model_name.endswith("-S") else "M"

            if "kitti" in model_name.lower():
                dataset_config = "kitti"
            elif "spring" in model_name.lower():
                dataset_config = "spring"
            else:
                dataset_config = "sintel"  # Fallback for generic Tartan models

            config_filename = f"{dataset_config}-{size}.json"

            # ---------------------------------------------------------
            # 2. Load Local JSON Configuration
            # ---------------------------------------------------------
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(current_dir, "sea-raft-config")
            config_path = os.path.join(config_dir, config_filename)

            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Configuration file missing!\n"
                    f"Expected to find: {config_path}\n"
                    f"Please ensure you placed the official config files (like {config_filename}) "
                    f"into the 'sea-raft-config' folder."
                )

            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            # ---------------------------------------------------------
            # 3. Download Model Checkpoint
            # ---------------------------------------------------------
            print(f"[SEA-RAFT Loader] Loading {model_name} from HuggingFace: {repo_id}")
            print(f"[SEA-RAFT Loader] Using configuration: {config_filename}")
            print("[SEA-RAFT Loader] First run downloads model, subsequent runs use cache...")

            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=None
            )
            print(f"[SEA-RAFT Loader] Downloaded to: {checkpoint_path}")

            # Load PyTorch checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # ---------------------------------------------------------
            # 4. Initialize Model with SafeNamespace + JSON Data
            # ---------------------------------------------------------
            args = SafeNamespace()

            # Base safety arguments in case the official JSON drops something obscure
            args.small = size == "S"
            args.mixed_precision = False
            args.alternate_corr = False
            args.dropout = 0.0
            args.pretrain = False

            # Dynamically apply all JSON keys into the args namespace
            for k, v in config_dict.items():
                setattr(args, k, v)

            # Initialize Model
            model = SEARAFT(args)
            model.load_state_dict(checkpoint)
            model = model.to(device).eval()

            # Cache the model
            cls._cache[cache_key] = model
            print(f"[SEA-RAFT Loader] ✓ Successfully loaded {model_name} on {device}")

            return model

        except Exception as e:
            raise RuntimeError(
                f"Failed to load SEA-RAFT model from HuggingFace or Local Config.\n\n"
                f"Model: {model_name}\n"
                f"Error: {e}\n\n"
                f"Troubleshooting:\n"
                f"1. Check that '{config_filename}' exists in the 'sea-raft-config' directory next to this script.\n"
                f"2. Check your internet connection\n"
                f"3. Update huggingface-hub: pip install --upgrade huggingface-hub\n\n"
                f"If the issue persists, report at:\n"
                f"https://github.com/cedarconnor/ComfyUI_MotionTransfer/issues"
            )

    @classmethod
    def clear_cache(cls):
        """Clear the model cache to free memory."""
        cls._cache.clear()
        print("[SEA-RAFT Loader] Cache cleared")


# Export public API
__all__ = ['SEARAFTLoader', 'SEARAFT']