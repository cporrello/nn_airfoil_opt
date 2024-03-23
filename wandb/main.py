from fcn import FullyConnectedNetwork
from train import train_and_plot
from plotter import plot_validation_cp_set
import pprint
import numpy as np
import torch

# Weights and Biases pipeline
import wandb
wandb.login(key="")

# Ensure reproducibility
torch.backends.cudnn.deterministic = True
seed_no = 216
np.random.seed(hash("improves reproducibility") % seed_no)
torch.manual_seed(hash("by removing stochasticity") % seed_no)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed_no)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# DATA LOADING
# ============================================================================== 

# Load data
x = np.load("../Geo-FNO/data/naca/NACA_Surface_x.npy")
y = np.load("../Geo-FNO/data/naca/NACA_Surface_y.npy")
p = np.load("../Geo-FNO/data/naca/NACA_Surface_p.npy")

# Reshape xy into single vector with [x_1 ... x_n, y_1, ..., y_n] for training
xy = np.concatenate((x, y), axis=1)

# Split data into training, validation and test sets
xy = torch.from_numpy(xy).float().to(device)
p = torch.from_numpy(p).float().to(device)
dataset = torch.utils.data.TensorDataset(xy, p)
splits = torch.utils.data.random_split(dataset, [0.80, 0.10, 0.10])
train_set, val_set, test_set = splits

# ==============================================================================
# WEIGHTS AND BIASES CONFIGURATION
# ==============================================================================

# WandB Hyperparameter dictionary
sweep_configuration = {
    "method": "grid",
    "name": "grid_search",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters":
    {
        "nlayers": {"values": [4, 6, 8]},
        "h_dim": {"values": [64, 128, 256]},
        "batch_size": {"values": [128, 256, 512]},
        "learning_rate": {"values": [1e-3, 1e-4, 1e-5]}
    }
}
pprint.pprint(sweep_configuration)
project_name = "project"
group_name = "CME216"
sweep_id = wandb.sweep(sweep_configuration, project=project_name)


# ==============================================================================
# DEFINE SWEEP LOOP
# ============================================================================== 

# Neural network parameters determined by data structure
input_dim = xy.shape[1]
output_dim = p.shape[1]

# Fixed Neural network parameters outside of sweep
activation = torch.nn.ReLU()
max_epochs = 5000

# Define training for each weights and biases run
def train(config=None):

    # Initialize new WandB rune
    run = wandb.init(config=config, project=project_name, group=group_name)
    config = wandb.config
    
    # Change sweep name for convenience
    name_str = f"nl{config.nlayers}_hd{config.h_dim}_bs{config.batch_size}_lr{config.learning_rate}"
    run.name = name_str

    # Training parameters
    nlayers = config.nlayers
    nunits = config.h_dim
    batch_size = config.batch_size
    learning_rate = config.learning_rate
   
    # Training with ADAM optimization
    fcn = FullyConnectedNetwork(input_dim, output_dim, nlayers, nunits, activation).to(device)
    optimizer = torch.optim.Adam(fcn.parameters(), lr=learning_rate)
    train_and_plot(fcn, optimizer, max_epochs, train_set, val_set, batch_size=batch_size)

    # At end of training, plot Cp distribution from validation set
    plot_validation_cp_set(fcn, val_set)

# Pipeline for WandB
wandb.agent(sweep_id, train)
wandb.finish()