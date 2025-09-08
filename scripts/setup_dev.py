#!/usr/bin/env python3
"""
Development environment setup script.
Automates common setup tasks for developers.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def main():
    """Set up the development environment."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🚀 Setting up Italian Teacher development environment...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install the package in development mode
    print("\n📦 Installing package in development mode...")
    try:
        run_command("pip install -e '.[dev,training,audio]'")
        print("✅ Package installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Full install failed, trying core dependencies only...")
        run_command("pip install -r requirements.txt")
        run_command("pip install -e .")
    
    # Install pre-commit hooks
    print("\n🔧 Setting up pre-commit hooks...")
    try:
        run_command("pre-commit install")
        print("✅ Pre-commit hooks installed")
    except subprocess.CalledProcessError:
        print("⚠️  Pre-commit not available, skipping hooks setup")
    
    # Create necessary directories
    print("\n📁 Creating necessary directories...")
    directories = [
        "data/raw",
        "data/processed", 
        "models/checkpoints",
        "models/adapters",
        "logs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    
    # Run initial tests
    print("\n🧪 Running initial tests...")
    try:
        run_command("python -m pytest tests/ -v", check=False)
    except subprocess.CalledProcessError:
        print("⚠️  Some tests failed, but setup continues...")
    
    print("\n🎉 Development environment setup complete!")
    print("\nNext steps:")
    print("1. Review configs/development.yaml")
    print("2. Run: python -m italian_teacher.cli --help")
    print("3. Start developing!")


if __name__ == "__main__":
    main()