import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

def split_imagenet_dataset(source_dir, output_dir, dataset_name, split_ratio=0.8):
    """
    Split ImageNet dataset into train/test folders.

    Args:
        source_dir: Path to source directory with class folders
        output_dir: Path where train/ and test/ folders will be created
        dataset_name: Name for logging (e.g., "ImageNet-1K", "ImageNet-R")
        split_ratio: Ratio for train (default 0.8 = 80% train, 20% test)
    """

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Splitting {dataset_name}")
    print(f"{'='*70}")
    print(f"Source: {source_dir}")
    print(f"Output train: {train_dir}")
    print(f"Output test: {test_dir}")
    print(f"Split ratio: {split_ratio:.0%} train / {1-split_ratio:.0%} test")
    print(f"{'='*70}\n")

    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(source_dir)
                        if os.path.isdir(os.path.join(source_dir, d))])

    print(f"Found {len(class_dirs)} classes")

    total_train = 0
    total_test = 0

    # Process each class
    for class_name in tqdm(class_dirs, desc=f"Processing {dataset_name}"):
        class_source_path = os.path.join(source_dir, class_name)

        # Get all images in this class
        images = [f for f in os.listdir(class_source_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

        if not images:
            continue

        # Shuffle and split
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(images))
        split_idx = int(len(images) * split_ratio)

        train_images = [images[i] for i in indices[:split_idx]]
        test_images = [images[i] for i in indices[split_idx:]]

        # Create class folders in train and test
        train_class_path = os.path.join(train_dir, class_name)
        test_class_path = os.path.join(test_dir, class_name)

        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        # Copy training images
        for img in train_images:
            src = os.path.join(class_source_path, img)
            dst = os.path.join(train_class_path, img)
            shutil.copy2(src, dst)
            total_train += 1

        # Copy test images
        for img in test_images:
            src = os.path.join(class_source_path, img)
            dst = os.path.join(test_class_path, img)
            shutil.copy2(src, dst)
            total_test += 1

    print(f"\n{'='*70}")
    print(f"✓ {dataset_name} split complete!")
    print(f"{'='*70}")
    print(f"Total train images: {total_train}")
    print(f"Total test images:  {total_test}")
    print(f"Ratio: {total_train}:{total_test} = {total_train/(total_train+total_test):.2%} train")
    print(f"{'='*70}\n")

    return {
        "dataset": dataset_name,
        "total_train": total_train,
        "total_test": total_test,
        "ratio": total_train / (total_train + total_test)
    }

def consolidate_imagenet100_splits(source_base, output_dir):
    """
    Consolidate ImageNet-100's 4-way train split into single train/test folders.

    Source structure:
        imagenet100/
            train.X1/  (classes with ~325 images each)
            train.X2/
            train.X3/
            train.X4/
            val.X/

    Output structure:
        imagenet100/
            train/  (merged from X1, X2, X3, X4)
            test/   (copy of val.X)
    """

    train_output = os.path.join(output_dir, 'train')
    test_output = os.path.join(output_dir, 'test')

    os.makedirs(train_output, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)

    print(f"\n{'='*80}")
    print("Consolidating ImageNet-100 4-way splits → train/test")
    print(f"{'='*80}\n")

    # Step 1: Merge train.X1, X2, X3, X4 into train/
    print("Merging training splits (train.X1, X2, X3, X4)...")
    total_train = 0

    for split_name in ['train.X1', 'train.X2', 'train.X3', 'train.X4']:
        split_dir = os.path.join(source_base, split_name)
        if not os.path.exists(split_dir):
            continue

        # Get all class folders in this split
        class_folders = [d for d in os.listdir(split_dir)
                        if os.path.isdir(os.path.join(split_dir, d))]



        for class_name in tqdm(class_folders, desc=f"Processing {split_name}"):
            class_source = os.path.join(split_dir, class_name)
            class_target = os.path.join(train_output, class_name)

            # Create class folder if doesn't exist
            os.makedirs(class_target, exist_ok=True)

            # Copy all images
            for img_file in os.listdir(class_source):
                img_path = os.path.join(class_source, img_file)
                if os.path.isfile(img_path):
                    dst_path = os.path.join(class_target, img_file)
                    shutil.copy2(img_path, dst_path)
                    total_train += 1

    print(f"✓ Training set: {total_train:,} images\n")

    # Step 2: Copy val.X to test/
    print("Copying validation set (val.X → test/)...")
    val_source = os.path.join(source_base, 'val.X')
    total_test = 0

    if os.path.exists(val_source):
        class_folders = [d for d in os.listdir(val_source)
                        if os.path.isdir(os.path.join(val_source, d))]

        for class_name in tqdm(class_folders, desc="Copying val.X"):
            class_source = os.path.join(val_source, class_name)
            class_target = os.path.join(test_output, class_name)

            os.makedirs(class_target, exist_ok=True)

            for img_file in os.listdir(class_source):
                img_path = os.path.join(class_source, img_file)
                if os.path.isfile(img_path):
                    dst_path = os.path.join(class_target, img_file)
                    shutil.copy2(img_path, dst_path)
                    total_test += 1

    print(f"✓ Test set: {total_test:,} images\n")

    print(f"{'='*80}")
    print(f"✓ Consolidation complete!")
    print(f"{'='*80}")
    print(f"Train images: {total_train:,}")
    print(f"Test images:  {total_test:,}")
    print(f"Ratio: {total_train/total_test:.1f}:1")
    print(f"\nOutput structure:")
    print(f"  {train_output}/")
    print(f"  {test_output}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Define paths based on your workspace structure
    BASE_PATH = r"d:\CODE\Class-Incremental-Learning-Code\Semantic_Grouping_with_Du_Supplementary Material"
    MY_DATA_PATH = os.path.join(BASE_PATH, "..", "..", "my_data")  # Adjust based on your actual my_data location

    MY_DATA_PATH = f"{BASE_PATH}\\my_data"
    os.makedirs(MY_DATA_PATH, exist_ok=True)


    # Split ImageNet-R
    imagenet_r_source = os.path.join(BASE_PATH, "ImageNet1R", "imagenet-r")
    imagenet_r_output = os.path.join(MY_DATA_PATH, "imagenet-r")  # For iImageNet_R class

    if os.path.exists(imagenet_r_source) and not os.path.exists(imagenet_r_output):
        stats_r = split_imagenet_dataset(imagenet_r_source, imagenet_r_output,
                                        "ImageNet-R", split_ratio=0.8)

    imagenet100_source = r"D:\CODE\Class-Incremental-Learning-Code\Semantic_Grouping_with_Du_Supplementary Material\ImageNet100"
    imagenet100_output = r"d:\CODE\Class-Incremental-Learning-Code\Semantic_Grouping_with_Du_Supplementary Material\my_data\imagenet100"

    consolidate_imagenet100_splits(imagenet100_source, imagenet100_output)
