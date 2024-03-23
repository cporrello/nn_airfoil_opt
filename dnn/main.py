from fcn import FullyConnectedNetwork
from train import train_and_plot
from plotter import plot_validation_cp_set
import numpy as np
import torch

# Ensure reproducibility
np.random.seed(216)
gen = torch.Generator().manual_seed(2023)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# DATA LOADING
# ============================================================================== 

# Load data
x = np.load("../Geo-FNO/data/naca/NACA_Surface_x.npy")
y = np.load("../Geo-FNO/data/naca/NACA_Surface_y.npy")
p = np.load("../Geo-FNO/data/naca/NACA_Surface_p.npy")

# Reshape xy into single vector with [x_1 ... x_n, y_1, ..., y_n] for training
xy = np.concatenate((x, y), axis=1)
xy = torch.from_numpy(xy).float().to(device)
p = torch.from_numpy(p).float().to(device)

# Geo-FNO uses last portion of data as the test set; manual splitting to match
i = 1000
j = i + 200
dataset = torch.utils.data.TensorDataset(xy, p)
train_set = torch.utils.data.Subset(dataset, range(i))
val_set = torch.utils.data.Subset(dataset, range(i, j))
test_set = torch.utils.data.Subset(dataset, range(j, len(p)))


# ==============================================================================
# TRAINING
# ============================================================================== 

# Training parameters
input_dim = xy.shape[1]
output_dim = p.shape[1]
nlayers = 6
nunits = 256
batch_size = 256
max_epochs = 2000
sigma = torch.nn.GELU()

# Training with ADAM optimization
fcn = FullyConnectedNetwork(input_dim, output_dim, nlayers, nunits, sigma).to(device)
optimizer = torch.optim.Adam(fcn.parameters(), lr=1e-3)
train_and_plot(fcn, optimizer, max_epochs, train_set, val_set, batch_size=batch_size)

# Plot validation
plot_validation_cp_set(fcn, val_set)

# Once finished training, save model
torch.save(fcn.state_dict(), "../models/test.pt")
