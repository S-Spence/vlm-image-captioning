import torch
import matplotlib.pyplot as plt
import evaluate
from pycocoevalcap.cider.cider import Cider
import textwrap
import pandas as pd
import os


def evaluate_bleu_cider(model, data_loader, display_captions=False, save_captions_path=None, max_batches=None, max_new_tokens=20,
                        do_sample=False, top_p=0.9, temperature=0.9, num_beams=2):
    """
    Evaluate the model using BLEU and CIDEr metrics.

    Parameters:
        model: VisionLanguageModel instance
        data_loader: DataLoader for evaluation
        display_captions (bool): if true, display predicted captions
        save_captions_path (str): path to save the displayed captions
        max_batches (int): maximum number of batches to evaluate
        max_new_tokens (int): maximum number of tokens to generate
        do_sample (bool): whether to sample captions
        top_p (float): nucleus sampling parameter
        temperature (float): sampling temperature
        num_beams (int): number of beams for beam search

    Returns:
        bleu_score (float): BLEU score
        cider_score (float): CIDEr score
    
    """
    predictions, targets, image_pred_pairs = {}, {}, []
    cider_metric = Cider()
    bleu_metric = evaluate.load("bleu")

    with torch.no_grad():
        sample_idx, batch_count = 0, 0
        for batch_images, batch_captions in data_loader:

            decoded_preds = model.predict(batch_images, max_new_tokens, 
                                          do_sample=do_sample, top_p=top_p, temperature=temperature,
                                          n_beams=num_beams)

            for prediction, reference, image in zip(decoded_preds, batch_captions, batch_images):
                # cider scorer expects two dictionaries with image_id as key and
                # list of captions as value. Prediction should always be length 1, as outlined in
                # doc string https://github.com/peteanderson80/coco-caption/blob/master/pycocoevalcap/cider/cider.py
                predictions[sample_idx] = [prediction]
                targets[sample_idx] = [ref for ref in reference]

                # collect six images to visualize a grid of captions if true
                if display_captions and len(image_pred_pairs) < 6:
                    image_pred_pairs.append((image, prediction))

                sample_idx += 1

            batch_count += 1

            if max_batches is not None and batch_count > max_batches:
                break

    if display_captions:
        title = "Experiment 1: "
        if do_sample:
            title += f"Sample Approach: Top-p: {top_p}, Temperature: {temperature}"
        else:
            title += f"Sample Approach: Num beams: {num_beams}"
        display_predicted_captions_grid(image_pred_pairs, save_captions_path, title)

    print(f"Sample Prediction: {predictions[0]}, Reference: {targets[0]}")
    bleu_references = [[ref for ref in targets[i]] for i in range(len(targets))]
    bleu_predictions = [predictions[i][0] for i in range(len(predictions))]
    bleu_score = bleu_metric.compute(predictions=bleu_predictions, references=bleu_references)["bleu"]
    cider_score, _ = cider_metric.compute_score(targets, predictions)

    return bleu_score, cider_score

def display_predicted_captions_grid(image_pred_pairs, save_captions_path=None, title=None):
    """
    Display a fixed 2x3 grid with 6 captions from image-prediction pairs.

    Parameters:
        image_pred_pairs (List[Tuple[Image, str]]): List of tuples containing images and their predicted captions.
        save_captions_path (str, optional): Path to save the plot image.
    """
    assert len(image_pred_pairs) >= 6
    
    pairs = image_pred_pairs[:6]
    
    _, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flat

    for i, ax in enumerate(axs):
        image, pred = pairs[i]
        ax.imshow(image)
        ax.axis("off")
        wrapped_caption = "\n".join(textwrap.wrap(pred, width=40))
        ax.set_title(wrapped_caption, fontsize=10, pad=10)

    if title:
        plt.suptitle(title, fontsize=16, y=1.02) 

    plt.tight_layout()
    if save_captions_path:
        plt.savefig(save_captions_path)
    plt.show()

def save_experiment_results(experiment, cider, bleu, results_path="results.csv"):
    """
    Save experiment results to a CSV file.
    Updates the file or writes a new file if it does not exist.

    Parameters:
        experiment (str): Name of the experiment
        cider (float): CIDEr score
        bleu (float): BLEU score
    """
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
    else:
        df = pd.DataFrame(columns=["experiment", "cider", "bleu"])

    new_row = {"experiment": experiment, "cider": round(cider, 4), "bleu": round(bleu, 4)}

    if experiment in df["experiment"].values:
        df.loc[df["experiment"] == experiment, ["cider", "bleu"]] = [cider, bleu]
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
