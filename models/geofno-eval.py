# Script used to evaluate post-training performance of trained neural networks
from geofno import FNO2d
from plotter import plot_test_cp_set_from_data
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
X = np.load("../Geo-FNO/data/naca/NACA_Cylinder_x.npy")
Y = np.load("../Geo-FNO/data/naca/NACA_Cylinder_y.npy")
Q = np.load("../Geo-FNO/data/naca/NACA_Cylinder_Q.npy")

# Convert vectors to PyTorch tensor arrays
X = torch.from_numpy(X).float().to(device)
Y = torch.from_numpy(Y).float().to(device)
Q = torch.from_numpy(Q).float().to(device)

# Inputs and Outputs for Geo-FNO
input = torch.stack([X, Y], dim=-1)
output = torch.tensor(Q[:, 3], dtype=torch.float).to(device)

# Geo-FNO uses last portion of data as the test set; manual splitting to match
i = int(0.8*len(output))
j = int(0.9*len(output))
dataset = torch.utils.data.TensorDataset(input, output)
test_set = torch.utils.data.Subset(dataset, range(j, len(Q)))



# ==============================================================================
# FULLY CONNECTED NETWORK
# ============================================================================== 

# Network design
modes = 12
width = 32
geofno = FNO2d(modes*2, modes, width).to(device)
geofno.load_state_dict(torch.load("geo-fno_80p_train_no_decay.pt"))
geofno.eval()

# Evaluate trained model on validation set
prediction = geofno(test_set.dataset.tensors[0][-3:])

# For plotting cp, need to extract the surface profile coordinates
cnx1 = 50
p = prediction.squeeze()
xs = test_set.dataset.tensors[0][-3:, cnx1:-cnx1, 0, 0]
ys = test_set.dataset.tensors[0][-3:, cnx1:-cnx1, 0, 1]
ps = p[:, cnx1:-cnx1, 0]

# Evaluate trained model on validation set
val_ps = test_set.dataset.tensors[1][-3:, cnx1:-cnx1, 0]
L2 = torch.nn.MSELoss()
err = L2(ps, val_ps)
print(f"Mean-squared error: {err:.4e}")
plot_test_cp_set_from_data(xs, ps, val_ps)
# """