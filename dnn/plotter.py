import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Ensure reproducibility
np.random.seed(216)

# LaTeX rendering
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def plot_validation_cp(model, val_set) -> None:

    # Just plot first element in training set
    idx = 0

    # Extract vectors for plotting
    val_cp_plot = val_set.dataset.tensors[1][idx]
    val_xy_plot = val_set.dataset.tensors[0][idx]

    # Convert to cpu for plotting
    pred_cp = model(val_xy_plot).detach().cpu().numpy()
    val_cp_plot = val_cp_plot.detach().cpu().numpy()
    val_xy_plot = val_xy_plot.detach().cpu().numpy()
    val_x = val_xy_plot[0:len(val_cp_plot)]

    # Plotting routine
    fig = plt.figure()
    plt.plot(val_x, val_cp_plot, label="Data")
    plt.plot(val_x, pred_cp, label="DNN")
    plt.xlabel(r"Chordwise position, $x/c$", fontsize=14)
    plt.ylabel(r"Coefficient of pressure, $c_p$", fontsize=14)
    plt.legend()
    plt.grid()

    # Flip axes to agree with convention
    plt.gca().invert_yaxis()


def plot_validation_cp_set(model, val_set) -> None:

    # Validation set is shuffled, so first six indices are random
    idxs = np.random.randint(0, high=len(val_set), size=6)
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    
    # Plotting routine
    k = 0
    for i in range(2):
        for j in range(3):

            # Extract vectors for plotting
            val_cp_plot = val_set.dataset.tensors[1][idxs[k]]
            val_xy_plot = val_set.dataset.tensors[0][idxs[k]]

            # Convert to cpu for plotting
            pred_cp = model(val_xy_plot).detach().cpu().numpy()
            val_cp_plot = val_cp_plot.detach().cpu().numpy()
            val_xy_plot = val_xy_plot.detach().cpu().numpy()
            val_x = val_xy_plot[0:len(val_cp_plot)]
    
            # Plotting routine
            axs[i, j].plot(val_x, cp(val_cp_plot))
            axs[i, j].plot(val_x, cp(pred_cp))
            axs[i, j].grid(which="both")
            
            # Flip axes to agree with Cp convention
            axs[i, j].invert_yaxis()
            
            # Next dataset
            k += 1

    # Shared plot settings for figure
    axs[1, 0].sharex(axs[0, 0])
    axs[1, 1].sharex(axs[0, 1])
    axs[1, 2].sharex(axs[0, 2])

    # Shared legend
    fig.legend([r"Validation Data", r"DNN Prediction"], 
               ncol=2, 
               loc="lower right",
               fontsize=14)
    fig.supxlabel(r"Chordwise position, $x/c$", fontsize=20)
    fig.supylabel(r"Coefficient of pressure, $C_p$", fontsize=20)


# Compute the pressure coefficient from pressure data (for zero alpha)
def cp(p, pinf=1.0, Minf=0.8, gamma=1.4):
    return (p-pinf)/(0.5*gamma*pinf*Minf**2.0)