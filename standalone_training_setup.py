"""
Standalone training setup for Colab that doesn't depend on main src package.
Use this to avoid import issues during training.
"""

import sys
from pathlib import Path


def setup_standalone_training():
    """Setup training environment without main src imports."""

    # Add fine_tuning directory to path
    project_root = Path.cwd()
    fine_tuning_path = project_root / "src" / "fine_tuning"

    if str(fine_tuning_path) not in sys.path:
        sys.path.insert(0, str(fine_tuning_path))

    print(f"✅ Added to Python path: {fine_tuning_path}")

    # Import training modules directly
    try:
        from config import get_default_config
        from inference import MarcoInference
        from lora_trainer import MarcoLoRATrainer

        print("✅ Training modules imported successfully")
        return MarcoLoRATrainer, get_default_config, MarcoInference
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the project root directory")
        return None, None, None


if __name__ == "__main__":
    setup_standalone_training()
