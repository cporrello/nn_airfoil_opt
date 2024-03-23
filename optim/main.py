import torch
import numpy as np
import matplotlib.pyplot as plt
from fcn import FullyConnectedNetwork
from naca import symmetric_NACA_4digit
from sdesign import sdesign
from force import get_aerodynamic_forces

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

# PyTorch model parameters from training (need to change if model changes)
input_dim = 121*2
output_dim = 121
nlayers = 6
nunits = 256
max_epochs = 5000
sigma = torch.nn.ELU()

# Load PyTorch model
model = FullyConnectedNetwork(input_dim, output_dim, nlayers, nunits, sigma).to(device)
model.load_state_dict(torch.load("../models/test.pt"))
model.eval()

# Generate NACA0012 coordinates clockwise from TE
ang = np.linspace(0, 1, output_dim, dtype=np.float64)
x_undeformed = 0.5 + 0.5*np.cos(2*np.pi*ang)
y_undeformed = symmetric_NACA_4digit(x_undeformed)

# Need airfoil (x, y) coordinate pairs in Torch tensor for optimization
x_undeformed = torch.tensor(x_undeformed, dtype=torch.float).to(device)
y_undeformed = torch.tensor(y_undeformed, dtype=torch.float).to(device)

# Setup ADAM optimization
theta = torch.zeros(7, dtype=torch.float, requires_grad=True, device="cuda")

# For some reason, setting maximize=True results in a minimum for the L/D ratio
optimizer = torch.optim.Adam([theta], lr=1E-5, maximize=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

# Begin optimization loop
for epoch in range(max_epochs):
    
    # Get predicted Cp ocurve from trained model
    x, y = sdesign(theta, x_undeformed, y_undeformed)
    xy = torch.cat((x, y))
    p = model(xy)

    # From design,get forces
    lift, drag = get_aerodynamic_forces(x, y, p)

    # Objective function
    loss = (drag/lift)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Print optimization status
    print(f"Epoch [{epoch + 1}/{max_epochs}]: {loss.item():.6f}")

# Display final optimal design
x_opt, y_opt = sdesign(theta, x_undeformed, y_undeformed)
x_opt, y_opt = x_opt.detach().cpu().numpy(), y_opt.detach().cpu().numpy()
fig, ax = plt.subplots()
ax.plot(x_opt, y_opt)
ax.set_aspect('equal')

# Save results for further post-processing
# np.savetxt("results/xs_fcn.csv", x_opt)
# np.savetxt("results/ys_fcn.csv", y_opt)