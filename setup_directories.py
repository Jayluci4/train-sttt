#!/usr/bin/env python3
"""
Setup script to create the directory structure for the unified training system.

This script creates all necessary directories and moves files to their proper locations.
"""

import os
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory(path, exist_ok=True):
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=exist_ok)
        logger.info(f"Created directory: {path}")
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")

def main():
    """Create the directory structure for the unified training system."""
    # Create main directories
    create_directory("configs")
    create_directory("data")
    create_directory("data/company_images")
    create_directory("output")
    create_directory("output/vision_phase1")
    create_directory("output/founder_vc_phase2")
    create_directory("output/complete_pipeline")
    create_directory("logs")
    
    # Create subdirectories for data
    create_directory("data/founders")
    create_directory("data/vcs")
    
    # Create placeholder files
    placeholder_content = "# This is a placeholder file. Replace with actual data."
    
    placeholder_files = [
        "data/company_dataset.csv",
        "data/company_images/README.md",
        "data/founders/README.md",
        "data/vcs/README.md"
    ]
    
    for file_path in placeholder_files:
        with open(file_path, 'w') as f:
            f.write(placeholder_content)
        logger.info(f"Created placeholder file: {file_path}")
    
    # Move configuration files to configs directory if they exist
    config_files = [
        "configs/phase1.json",
        "configs/phase2.json"
    ]
    
    for file_path in config_files:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                if "phase1" in file_path:
                    f.write('{\n  "model_name": "microsoft/Phi-3.5-vision-instruct",\n  "output_dir": "./output/vision_phase1"\n}')
                else:
                    f.write('{\n  "model_name": "microsoft/Phi-3.5-mini",\n  "output_dir": "./output/founder_vc_phase2"\n}')
            logger.info(f"Created basic config file: {file_path}")
    
    # Move founder data files if they exist
    founder_files = [
        "1000 Founder Profiles.json",
        "Founder-VC Match Pairs 101-200.markdown"
    ]
    
    for file_path in founder_files:
        if os.path.exists(file_path):
            target_path = os.path.join("data", file_path)
            shutil.move(file_path, target_path)
            logger.info(f"Moved {file_path} to {target_path}")
    
    logger.info("Directory setup complete!")

if __name__ == "__main__":
    main() 