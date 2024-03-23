import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Ensure reproducibility
np.random.seed(216)

# LaTeX rendering
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def plot_test_cp(model, test_set) -> None:

    # Just plot first element in training set
    idx = 0

    # Extract vectors for plotting
    test_cp_plot = test_set.dataset.tensors[1][idx]
    test_xy_plot = test_set.dataset.tensors[0][idx]

    # Convert to cpu for plotting
    pred_cp = model(test_xy_plot).detach().cpu().numpy()
    test_cp_plot = test_cp_plot.detach().cpu().numpy()
    test_xy_plot = test_xy_plot.detach().cpu().numpy()
    test_x = test_xy_plot[0:len(test_cp_plot)]

    # Plotting routine
    fig = plt.figure()
    plt.plot(test_x, test_cp_plot, label="Data")
    plt.plot(test_x, pred_cp, label="DNN")
    plt.xlabel(r"Chordwise position, $x/c$", fontsize=14)
    plt.ylabel(r"Coefficient of pressure, $c_p$", fontsize=14)
    plt.legend()
    plt.grid()

    # Flip axes to agree with convention
    plt.gca().invert_yaxis()


def plot_test_cp_set_from_model(model, test_set) -> None:

    # test set is shuffled, so we just use first 3 indices
    idxs = [-3, -2, -1]
    fig, axs = plt.subplots(1, 3, figsize=(30, 8), layout="constrained")
    
    # Plotting routine
    for i in range(len(idxs)):

        # Extract vectors for plotting
        test_cp_plot = test_set.dataset.tensors[1][idxs[i]]
        test_xy_plot = test_set.dataset.tensors[0][idxs[i]]

        # Convert to cpu for plotting
        pred_cp = model(test_xy_plot).detach().cpu().numpy()
        test_cp_plot = test_cp_plot.detach().cpu().numpy()
        test_xy_plot = test_xy_plot.detach().cpu().numpy()
        test_x = test_xy_plot[0:len(test_cp_plot)]

        # Plotting routine
        axs[i].plot(test_x, cp(test_cp_plot))
        axs[i].plot(test_x, cp(pred_cp))
        axs[i].grid(which="both")
        
        # Flip axes to agree with Cp convention
        axs[i].invert_yaxis()
        

    # Shared legend
    fig.legend([r"Test Dataset", r"DNN Prediction"], 
               ncol=2, 
               loc="lower right",
               fontsize=18)
    fig.supxlabel(r"Chordwise position, $x/c$", fontsize=30)
    fig.supylabel(r"Pressure coefficient, $C_p$", fontsize=30)

    # Save figure at end
    plt.savefig("test_set_performance.pdf", format="pdf", bbox_inches="tight")


def plot_test_cp_set_from_data(xs, ps, test_p) -> None:
    
    # Plotting routine
    fig, axs = plt.subplots(1, 3, figsize=(30, 8), layout="constrained")
    for i in range(len(ps)):

        # Convert to cpu for plotting
        test_p_plot = test_p.detach().cpu().numpy()
        ps_plot = ps.detach().cpu().numpy()
        xs_plot = xs.detach().cpu().numpy()

        # Plotting routine
        axs[i].plot(xs_plot[i], cp(ps_plot[i]))
        axs[i].plot(xs_plot[i], cp(test_p_plot[i]))
        axs[i].grid(which="both")
        
        # Flip axes to agree with Cp convention
        axs[i].invert_yaxis()
        

    # Shared legend
    fig.legend([r"Test Dataset", r"DNN Prediction"], 
               ncol=2, 
               loc="lower right",
               fontsize=18)
    fig.supxlabel(r"Chordwise position, $x/c$", fontsize=30)
    fig.supylabel(r"Pressure coefficient, $C_p$", fontsize=30)

    # Save figure at end
    plt.savefig("test_set_performance.pdf", format="pdf", bbox_inches="tight")


# Compute the pressure coefficient from pressure data (for zero alpha)
def cp(p, pinf=1.0, Minf=0.8, gamma=1.4):
    return (p-pinf)/(0.5*gamma*pinf*Minf**2.0)