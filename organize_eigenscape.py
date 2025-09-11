#!/usr/bin/env python3
"""
Script to organize eigenscape files into folders based on filename patterns.
"""

import os
import shutil
from pathlib import Path
import argparse

def organize_eigenscape_files(source_dir: str, target_dir: str = None, dry_run: bool = False):
    """
    Organize eigenscape files into folders based on filename patterns.
    
    Args:
        source_dir: Directory containing the raw eigenscape files
        target_dir: Directory to create organized folders (default: source_dir + '_organized')
        dry_run: If True, only print what would be done without actually moving files
    """
    source_path = Path(source_dir)
    if target_dir is None:
        target_dir = str(source_path.parent / f"{source_path.name}_organized")
    target_path = Path(target_dir)
    
    # Define the mapping from filename patterns to folder names
    folder_mapping = {
        'Beach': 'Beach',
        'BusyStreet': 'Busy Street', 
        'Park': 'Park',
        'PedestrianZone': 'Pedestrian Zone',
        'QuietStreet': 'Quiet Street',
        'ShoppingCentre': 'Shopping Centre',
        'TrainStation': 'Train Station',
        'Woodland': 'Woodland'
    }
    
    # Create target directories
    for folder_name in folder_mapping.values():
        folder_path = target_path / folder_name
        if not dry_run:
            folder_path.mkdir(parents=True, exist_ok=True)
        print(f"{'Would create' if dry_run else 'Created'} directory: {folder_path}")
    
    # Process files
    moved_count = 0
    for file_path in source_path.glob("*.wav"):
        filename = file_path.name
        
        # Find matching folder pattern
        matched_folder = None
        for pattern, folder_name in folder_mapping.items():
            if pattern in filename:
                matched_folder = folder_name
                break
        
        if matched_folder:
            target_file_path = target_path / matched_folder / filename
            if not dry_run:
                shutil.copy2(file_path, target_file_path)
            print(f"{'Would move' if dry_run else 'Moved'}: {filename} -> {matched_folder}/")
            moved_count += 1
        else:
            print(f"Warning: No folder pattern matched for {filename}")
    
    print(f"\n{'Would move' if dry_run else 'Moved'} {moved_count} files total")
    
    if dry_run:
        print("\nThis was a dry run. Use --no-dry-run to actually organize the files.")
    else:
        print(f"\nFiles organized into: {target_path}")
        print("You can now use this organized directory with the dataloader.")

def main():
    parser = argparse.ArgumentParser(description="Organize eigenscape files into folders")
    parser.add_argument("source_dir", help="Directory containing eigenscape files")
    parser.add_argument("--target-dir", help="Target directory for organized files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually moving files")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually move the files (default is dry run)")
    
    args = parser.parse_args()
    
    # Default to dry run unless explicitly disabled
    dry_run = args.dry_run or not args.no_dry_run
    
    organize_eigenscape_files(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        dry_run=dry_run
    )

if __name__ == "__main__":
    main()
