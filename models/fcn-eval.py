# Script used to evaluate post-training performance of trained neural networks
from fcn import FullyConnectedNetwork
from plotter import plot_test_cp_set_from_model
import numpy as np
import torch

# Reproducibility
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

# Split data into training, validation and test sets
xy = torch.from_numpy(xy).float().to(device)
p = torch.from_numpy(p).float().to(device)
dataset = torch.utils.data.TensorDataset(xy, p)

# Geo-FNO uses last portion of data as the test set; manual splitting to match
i = int(0.8*len(p))
j = int(0.9*len(p))
dataset = torch.utils.data.TensorDataset(xy, p)
test_set = torch.utils.data.Subset(dataset, range(j, len(p)))



# ==============================================================================
# FULLY CONNECTED NETWORK
# ============================================================================== 

# Training parameters
input_dim = xy.shape[1]
output_dim = p.shape[1]
nlayers = 6
nunits = 256
batch_size = 256
max_epochs = 5000
sigma = torch.nn.ELU()

# Load pre-trained model
fcn = FullyConnectedNetwork(input_dim, output_dim, nlayers, nunits, sigma).to(device)
fcn.load_state_dict(torch.load("fcn_elu_2.pt"))
fcn.eval()

# Evaluate trained model on validation set
prediction = fcn(test_set.dataset.tensors[0])
L2 = torch.nn.MSELoss()
err = L2(prediction, test_set.dataset.tensors[1])
print(f"Mean-squared error: {err:.4e}")

# For Geo-FNO, we use only the last 3 points from the test set
plot_test_cp_set_from_model(fcn, test_set)