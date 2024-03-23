import torch
import torch.nn as nn
import matplotlib
import numpy as np
import wandb
import os
from error import rel_l2_error_on_set

# Can't generate figures in GUI mode during WandB sweeps
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_and_plot(model, optimizer, max_epochs, train_set, val_set, batch_size=1):

    # Statistics to keep track of
    loss_list = []
    val_loss_list = []
    print_freq = int(0.10*max_epochs)
    log_freq = 1

    # Construct training and validation loaders to deal with batched training
    training_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True
    )
    validation_inp = val_set.dataset.tensors[0]
    validation_out = val_set.dataset.tensors[1]

    # Criterion 
    criterion = nn.MSELoss()

    for epoch in range(max_epochs):
        for (train_x, train_y) in training_loader:
            
            # Forward pass to compute values
            model.train()
            optimizer.zero_grad()
            prediction = model(train_x)
            loss = criterion(prediction, train_y)

            # Backwards pass and adjust weights
            loss.backward()
            optimizer.step()

        # Training statistics to be printed
        if (epoch + 1) % log_freq == 0:

            # Store training loss
            loss_list.append(loss.item())    

            # Compute validation loss
            model.eval()
            with torch.no_grad():
                prediction = model(validation_inp)
                val_loss = criterion(prediction, validation_out)
            val_loss_list.append(val_loss.item())
            
            # Compute mean relative error for the validation set
            sample_rel_error = rel_l2_error_on_set(prediction, validation_out).detach().cpu().numpy()
            mean_rel_error = np.mean(sample_rel_error)

            # Log values to Weights and Biases
            wandb.log({"val_loss": val_loss.item(), 
                       "train_loss": loss.item(), 
                       "mean_relative_error": mean_rel_error})

            # Print training status
            if (epoch + 1) % print_freq == 0:
                print(f'Epoch [{epoch + 1}/{max_epochs}], Training Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}, Mean Relative Validation Error: {mean_rel_error:07.4f}%')
        
    # At end of training, save model to potentially import in the future
    save_path = os.path.join(wandb.run.dir, "model.ckpt")
    torch.save(model.state_dict(), save_path)

    # When here, training has concluded and plot training curves
    epoch_vector = np.arange(log_freq, max_epochs+log_freq, log_freq, dtype=np.int64)
    fig = plt.figure()
    plt.semilogy(epoch_vector, loss_list, label="Training set")
    plt.semilogy(epoch_vector, val_loss_list, label="Validation set")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Save figure and send to Weights and Biases log
    wandb.log({"Loss History": fig})