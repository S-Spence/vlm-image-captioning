import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import data_processing as dp
import numpy as np
import evaluation as eval

def train(model, train_dir, val_dir, loss_function, device, learning_rate=2e-5,
          lr_scheduler=False, batch_size=4, num_epochs=3, log_interval=100, max_batches=10000,
          training_type="mlp-pretrain", mlp_weights_path=None, model_weights_path="model_weights.pt", 
          loss_plot_path="training_loss.png", eval_every=1, random_seed=1, all_epochs_loss_plot_path="all_epochs_loss.png"):
    """
    Trains or fine-tunes a Vision-Language model.

    Parameters:
        model: VisionLanguageModel instance
        train_dir (str): path to training directory
        val_dir (str): path to validation directory
        loss_function (object): loss function
        device (str): 'cuda', 'mps', or 'cpu'
        learning_rate (float): learning rate
        lr_scheduler (bool): whether to use a learning rate scheduler
        batch_size (int): number of images per batch
        num_epochs (int): number of training epochs
        log_interval (int): how often to log and plot training loss
        max_batches (int): maximum number of batches to train on
        training_type (str): 'mlp-pretrain', 'sft', or 'lora'
        mlp_weights_path (str): path to save or load MLP weights
        model_weights_path (str): path to save the model weights
        loss_plot_path (str): where to save the training loss plot
        eval_every (int): how often to evaluate BLEU and CIDEr scores
        random_seed (int): random seed for reproducibility in data sampling
        all_epochs_loss_plot_path: the path to save the avg loss over epochs plot
    """
    all_epoch_losses, all_epoch_images = [], []

    # load a pretrained mlp if fine tuning for llava stg 2
    if (training_type == "sft" or training_type == "lora") and mlp_weights_path is not None:
        print(f"Loading MLP weights from {mlp_weights_path}")
        model.image_encoder.projection.load_state_dict(torch.load(mlp_weights_path, map_location=device))

    # unfreeze the model parameters based on the training type provided
    unfreeze_params_by_train_type(training_type, model)

    # define the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    if lr_scheduler:
        # define the learning rate scheduler
        total_steps = max_batches * num_epochs
        learning_rate_scheduler = cosine_with_warmup_lr_scheduler(optimizer, total_steps, warmup_steps=200)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        # create the validation and training loaders for each epoch
        # so we will not run out of data in later epics when it reaches the end of the dataset
        # additionally, this ensures it sees the same training data per epoch
        val_loader = dp.batch_stream("captions.txt", val_dir, batch_size=batch_size, eval_mode=True, seed=random_seed)
        train_loader = dp.batch_stream("captions.txt", train_dir, batch_size=batch_size, eval_mode=False, seed=random_seed)

        running_loss = 0.0
        step_losses, images_seen = [], []
        cumulative_images = 0

        model.train()

        # intialize prefix and get tokenized length 
        prefix = "Caption: "
        prefix_ids = model.tokenizer(prefix, return_tensors="pt").input_ids[0]
        prefix_len = prefix_ids.size(0)

        for step, (images, captions) in enumerate(train_loader):
            if step >= max_batches:
                break

            if model.cross_attention:
                model.tokenizer.padding_side = 'left'
                full_captions = [caption[0] for caption in captions]
                tokenized = model.tokenizer(full_captions, return_tensors="pt", padding=True, truncation=True).to(device)
                input_ids = tokenized.input_ids

                # build shifted inputs and targets
                bos = torch.full((input_ids.size(0), 1), model.tokenizer.bos_token_id, dtype=torch.long, device=device)
                inputs = torch.cat([bos, input_ids[:, :-1]], dim=1) 
                targets = input_ids

            else:
                # structure prompts for the model
                # in training mode, captions will always be a list with one caption
                # because images will be included again with the other captions
                prompts = [f"Caption: {caption[0]}" for caption in captions]
                input_ids = model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]

            optimizer.zero_grad()

            logits = model(images, inputs)

            logits = logits.transpose(1, 2)

            # remove the prefix image tokens only if cross attention is not used
            num_image_tokens = model.image_encoder.num_image_tokens if not model.cross_attention else 0
            # drop the image embedding and only calculate loss of the caption
            logits = logits[:, :, num_image_tokens:]

            # also drop the Caption: prefix from loss calculations for the 
            # non-cross attention case
            if not model.cross_attention:
                logits = logits[:, :, prefix_len:]
                targets = targets[:, prefix_len:]

            loss = loss_function(logits, targets)

            loss.backward()

            # clip the gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            if lr_scheduler:
                learning_rate_scheduler.step()

            running_loss += loss.item()
            cumulative_images += len(images)

            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                print(f"Step {step + 1}: Loss = {avg_loss:.4f}")
                step_losses.append(avg_loss)
                images_seen.append(cumulative_images)
                running_loss = 0.0

                plot_running_loss(all_epoch_losses + [step_losses], all_epoch_images + [images_seen], loss_plot_path)

        all_epoch_losses.append(step_losses)
        all_epoch_images.append(images_seen)

        # only evaluate BLEU and CIDEr if not pretraining the MLP
        # this is because the MLP is trained to optimize image embeddings
        # rather than produce meaningful captions
        if training_type != "mlp-pretrain" and (epoch+1) % eval_every == 0:
            bleu, cider = eval.evaluate_bleu_cider(model, val_loader, display_captions=False, max_batches=100, max_new_tokens=15,
                                                   do_sample=False, num_beams=2)
            print(f"Epoch {epoch + 1} - BLEU: {bleu:.4f}, CIDEr: {cider:.4f}")

    # save MLP weights if pretraining
    if training_type == "mlp-pretrain" and mlp_weights_path is not None:
        torch.save(model.image_encoder.projection.state_dict(), mlp_weights_path)
        print(f"MLP weights saved to {mlp_weights_path}")

    # plot loss over epochs for fine tuning and sabe the decoder wieghts
    if training_type == "sft" or training_type == "lora":
        plot_and_save_epoch_loss(all_epoch_losses, save_path=all_epochs_loss_plot_path)
        # save model
        torch.save(model.state_dict(), model_weights_path)
        print(f"Decoder model saved to {model_weights_path}")

    print("\nTraining complete.")

def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    """
    Constructs a cosine learning rate scheduler with warmup.

    Note: function provided in class assignment

    Parameters:
        opt: torch optimizer
        total_steps (int): total number of training steps
        warmup_steps (int): number of warmup steps

    Returns:
        scheduler: torch.optim.lr_scheduler
    """
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum)/float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum-warmup_steps
            tail_steps = total_steps-warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592*prog)+1.0)*0.5)*0.9 + 0.1
        return max(lrmult, 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)
    return scheduler

def unfreeze_params_by_train_type(training_type, model):
    """
    This function unfreezes the parameters of the model based on training type.
        - sft: unfreezes the whole decoder and leaves the mlp trainable
        - mlp-pretrain: only trains the mlp
        - lora: adds lora adapters and freezes the rest of the decoder, leaving the mlp trainable
    
    Parameters:
        training_type (str): 'mlp-pretrain', 'sft', or 'lora'
        model: VisionLanguageModel instance
    """
    # pretrain the img projection mlp for llava stg 1
    if training_type == "mlp-pretrain":
        # nothing to do, the mlp is already trainable
        print("Encoder and decoder weights frozen for LLava-style MLP pre-training.")
        
    # fine-tune the decoder and mlp for llava stg 2
    elif training_type == "sft":
        print("Fine-tuning full decoder and MLP.")
        for param in model.decoder.parameters():
            param.requires_grad = True
    
    elif training_type == "lora":
        print("fine tuning LoRA Adapters and MLP")
        # add lora adapters and freeze the rest of the decoder
        model.decoder.add_lora()
    
def plot_running_loss(all_epoch_losses, all_image_counts, save_path="training_loss_by_images.jpg"):
    """
    Plot and save the training loss over epochs.

    Parameters:
        all_epoch_losses (List[List[float]]): List of epoch losses.
        all_image_counts (List[List[int]]): List of image counts seen.
        save_path (str): Path to save the plot image.
    """
    num_epochs = len(all_epoch_losses)
    colors = cm.get_cmap('tab10', num_epochs)

    plt.figure(figsize=(10, 6))
    for epoch_idx, (epoch_losses, image_counts) in enumerate(zip(all_epoch_losses, all_image_counts)):
        plt.plot(image_counts, epoch_losses, label=f"Epoch {epoch_idx + 1}", marker="o", color=colors(epoch_idx))
    plt.title("Training Loss vs. Images Seen")
    plt.xlabel("Images Seen")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_and_save_epoch_loss(running_avg_losses, save_path="loss_over_epochs.jpg"):
    """
    Compute, plot, and save the average loss over epochs.
    
    Parameters:
        running_avg_losses (list of list of float): Each inner list contains the running 
            average loss values recorded at evaluation steps within an epoch.
        save_path (str, optional): File path to save the plot image. Defaults to "loss_over_epochs.png".
    """
    # compute the average loss per epoch
    epoch_avg_losses = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in running_avg_losses]
    
    epochs = range(1, len(epoch_avg_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, epoch_avg_losses, marker='o', linestyle='-', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss over Epochs')
    plt.grid(True)
    
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")
    plt.close()
