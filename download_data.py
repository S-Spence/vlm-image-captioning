import os
import zipfile
import random
import shutil
import csv
from io import StringIO

def download_dataset(output_dir):
    """
    Downloads the Flickr30k dataset using curl.
    
    Parameters:
      output_dir (str): directory where the dataset will be stored.
    
    Returns:
      str: path to the downloaded zip file.
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_filename = os.path.join(output_dir, "flickr30k.zip")
    
    print("Downloading Flickr30k dataset...")
    curl_command = (
        f'curl -L -o {zip_filename} '
        'https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr30k'
    )
    os.system(curl_command)
    print("Download complete.")
    return zip_filename

def extract_dataset(zip_filename, output_dir):
    """
    Extracts the dataset zip file into the output directory and removes the zip.
    
    Parameters:
      zip_filename (str): path to the zip file.
      output_dir (str): directory where the dataset is extracted.
    """
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("Extraction complete.")
    os.remove(zip_filename)

def partition_images(output_dir, train_ratio=0.8, seed=42):
    """
    Partitions the images into train, validation, and test directories.
    
    Parameters:
      output_dir (str): the root directory where the dataset is extracted.
      seed (int): random seed for reproducible splits.
    
    Returns:
      tuple: Three sets containing filenames for train, val, and test splits.
    """
    print("Partitioning images...")
    images_dir = os.path.join(output_dir, "Images")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Expected images folder not found in {output_dir}")
    
    # create directories for each partition.
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # gather image files.
    image_files = sorted([f for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    random.seed(seed)
    random.shuffle(image_files)
    
    n_images = len(image_files)
    train_size = int(train_ratio * n_images)
    remainder = n_images - train_size
    # split the remainder equally between validation and test.
    val_size = remainder // 2
    
    train_files = set(image_files[:train_size])
    val_files = set(image_files[train_size:train_size + val_size])
    test_files = set(image_files[train_size + val_size:])
    
    # copy files into partition folders
    for f in train_files:
        shutil.copy(os.path.join(images_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy(os.path.join(images_dir, f), os.path.join(val_dir, f))
    for f in test_files:
        shutil.copy(os.path.join(images_dir, f), os.path.join(test_dir, f))
    
    print(f"Partitioning complete: {len(train_files)} training images, "
          f"{len(val_files)} validation images, {len(test_files)} test images.")
    return train_files, val_files, test_files

def partition_captions(output_dir, train_files, val_files, test_files):
    """
    Partitions the captions based on the image splits.
    
    Parameters:
      output_dir (str): root directory where the dataset is extracted.
      train_files, val_files, test_files (set): sets of image filenames for each split.
    """
    print("Partitioning captions...")
    captions_file = os.path.join(output_dir, "captions.txt")
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Expected captions.txt not found in {output_dir}")
    
    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    header = lines[0]
    data_lines = lines[1:]
    
    train, val, test = [], [], []
    
    for line in data_lines:
        reader = csv.reader(StringIO(line))
        for row in reader:
            if len(row) != 2:
                # skip lines with missing data
                continue

            image_file = row[0].strip()
            if image_file in train_files:
                train.append(line)
            elif image_file in val_files:
                val.append(line)
            elif image_file in test_files:
                test.append(line)
    
    # save the captions in each partition's directory.
    train_caption_path = os.path.join(output_dir, "train", "captions.txt")
    val_caption_path   = os.path.join(output_dir, "val", "captions.txt")
    test_caption_path  = os.path.join(output_dir, "test", "captions.txt")
    
    with open(train_caption_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(train)
    
    with open(val_caption_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(val)
    
    with open(test_caption_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(test)
    

def download_and_partition(output_dir="./flickr30k_data"):
    """
    Download and partition the dataset into train, validation, and test sets.

    Parameters:
        output_dir (str): directory where the dataset will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    zip_filename = download_dataset(output_dir)
    extract_dataset(zip_filename, output_dir)
    
    train_files, val_files, test_files = partition_images(output_dir, seed=42)
    partition_captions(output_dir, train_files, val_files, test_files)

    # cleanup the unpartitioned files
    images_dir = os.path.join(output_dir, "Images")
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    
    captions_file = os.path.join(output_dir, "captions.txt")
    if os.path.exists(captions_file):
        os.remove(captions_file)
    
    print("Dataset preparation complete.")

if __name__ == '__main__':
    download_and_partition()
