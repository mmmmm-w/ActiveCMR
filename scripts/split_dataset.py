import os
import shutil
import random
from pathlib import Path
import argparse

def create_split_dirs(base_dir):
    """Create train, val, and test directories if they don't exist."""
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)

def get_subject_dirs(dataset_dir):
    """Get list of subject directories."""
    return [d for d in os.listdir(dataset_dir) 
            if os.path.isdir(os.path.join(dataset_dir, d))]

def copy_subject(src_dir, dst_dir, subject):
    """Copy a subject directory to the destination."""
    src_path = os.path.join(src_dir, subject)
    dst_path = os.path.join(dst_dir, subject)
    shutil.copytree(src_path, dst_path)

def split_dataset(dataset_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed for reproducibility
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set random seed
    random.seed(seed)
    
    # Get all subject directories
    subjects = get_subject_dirs(dataset_dir)
    random.shuffle(subjects)
    
    # Calculate split sizes
    n_subjects = len(subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    
    # Split subjects
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]
    
    # Create split directories
    create_split_dirs(dataset_dir)
    
    # Copy files to respective directories
    splits = {
        'train': train_subjects,
        'val': val_subjects,
        'test': test_subjects
    }
    
    for split_name, split_subjects in splits.items():
        split_dir = os.path.join(dataset_dir, split_name)
        print(f"\nCopying {len(split_subjects)} subjects to {split_name} set...")
        for subject in split_subjects:
            copy_subject(dataset_dir, split_dir, subject)
            print(f"Copied {subject} to {split_name}")

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train, validation, and test sets')
    parser.add_argument('--dataset_dir', type=str, default='Dataset',
                      help='Path to the dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                      help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                      help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                      help='Ratio of test data')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Convert to absolute path
    dataset_dir = os.path.abspath(args.dataset_dir)
    
    print(f"Splitting dataset in {dataset_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Random seed: {args.seed}")
    
    split_dataset(
        dataset_dir=dataset_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

if __name__ == '__main__':
    main() 