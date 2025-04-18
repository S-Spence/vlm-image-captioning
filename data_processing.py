import os
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
import textwrap

def batch_stream(captions_file, image_dir, batch_size=8, seed=None, eval_mode=False):
    """
    Streams batches of (image, caption[s]) pairs from disk.

    Parameters:
        - captions_file (str): path to the captions file
        - image_dir (str): directory where images are stored
        - batch_size (int): number of images per batch
        - seed (int): random seed for reproducibility
        - eval_mode (bool): if True, returns all captions for each image
                            if False, returns a single image-caption pair with duplicate imgs

    Returns:
        - Tuple[List[PIL.Image.Image], List[List[str]]]: each image has a list of one or more captions
                                                         depending on mode 
    """
    entries = (load_captions_multi(captions_file, image_dir)
        if eval_mode else load_captions_single(captions_file, image_dir))

    if seed is not None:
        random.seed(seed)
    random.shuffle(entries)

    batch_images, batch_captions = [], []

    for image_name, captions in entries:
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            batch_images.append(image)
            batch_captions.append(captions)
        except Exception:
            continue

        if len(batch_images) == batch_size:
            yield batch_images, batch_captions
            batch_images, batch_captions = [], []

    if batch_images:
        yield batch_images, batch_captions

def load_captions_single(captions_file, image_dir):
    """
    Loads (image, single_caption) pairs — used for training (1:1 mapping)
    Parameters:
        captions_file: path to the captions file
        image_dir: directory where images are stored

    Returns: List of (image_name, [caption])
    """
    entries = []
    with open(os.path.join(image_dir, captions_file), "r", encoding="utf-8") as file:
        next(file) 
        for line in file:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                image_name, caption = parts
                entries.append((image_name, [caption]))
    return entries

def load_captions_multi(captions_file, image_dir):
    """
    Loads all captions per image — used for evaluation (1:N mapping)

    Parameters:
        captions_file: path to the captions file
        image_dir: directory where images are stored

    Returns: List of (image_name, [caption1, caption2, ...])
    """
    caption_dict = {}
    with open(os.path.join(image_dir, captions_file), "r", encoding="utf-8") as file:
        next(file)
        for line in file:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                image_name, caption = parts
                caption_dict.setdefault(image_name, []).append(caption)
    return list(caption_dict.items())

def visualize_random_captions(images_and_captions, n_images=6, figsize=(12, 8), seed=None):
    """
    Displays n random image-caption pairs from a list of (image, caption) batches.

    Parameters:
        images_and_captions: List of batches containing (images, captions)
        n_images: Number of images to visualize
        figsize: Size of the matplotlib figure
        seed: Optional random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    # flatten all batches into a single list of (image, caption) pairs
    flattened = [(img, cap[0]) for imgs, caps in images_and_captions for img, cap in zip(imgs, caps)]
    image_count = len(flattened)
    assert image_count >= n_images, f"Only {image_count} images available."

    # randomly sample images
    sampled = random.sample(flattened, n_images)

    # determine rows and columns for the grid
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols

    _, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for ax, (image, caption) in zip(axes, sampled):
        ax.imshow(image)
        ax.axis("off")
        wrapped_caption = "\n".join(textwrap.wrap(caption, width=40))
        ax.set_title(wrapped_caption, fontsize=10)

    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
    
    
