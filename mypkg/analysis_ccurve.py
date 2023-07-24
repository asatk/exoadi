import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import redux_utils

COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

def plot_ccurve(x: np.ndarray, y: np.ndarray, algo: str,
        save_path: str=None, fig_ax: tuple[plt.Figure, plt.Axes]=None,
        **kwargs):
    
    color = kwargs.get("color", None)
    xlabel = kwargs.get("xlabel", None)
    ylabel = kwargs.get("ylabel", None)
    title = kwargs.get("title", "Constrast Curve")

    xmax = np.max(x)
    xend = x[-1]
    yend = y[-1]
    xlims = (0.0, np.max(x))

    y_in_unit_intv = np.max(y) <= 1.0 and np.min(y) >= 0.0
    if y_in_unit_intv:
        ylims = (0.0, 1.0)
    else:
        ylims = (-0.1, 1.1)

    fig, ax = plt.subplots(figsize=(8, 6)) if fig_ax is None else fig_ax

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.plot(x, y, label=algo, color=color, alpha=0.75)
    ax.annotate(text=algo, xy=[xend, yend], xytext=[xend-0.055, yend-0.05], color=color)
    if not y_in_unit_intv:
        ax.hlines([0.0, 1.0], xmin=0.0, xmax=xmax, linestyles="dashed", colors="gray", alpha=0.75)
    
    if save_path is not None:
        fig.savefig(save_path)
    
    return fig, ax

if __name__ == "__main__":

    use_arcsec = True
    scale = "arcsec" if use_arcsec else "px"
    path = "out/datafr_vipPCA003-{:s}_45-74_skip20.csv"
    save_path = "cc_vipPCA003-{:s}_45-74_skip20.png"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    algos = ["single", "double", "annular"]

    xkey = "distance_arcsec" if use_arcsec else "distance"
    xlabel = f"Distance ({scale})"
    # ykey = "throughput"
    # ykey = "sensitivity_gaussian"
    ykey = "sensitivity_student"
    # ylabel = "Throughput"
    # ylabel = "Gaussian Sensitivity (Constrast)"
    ylabel = "Student Sensitivity (Small-Statistics Constrast)"

    cols = [xkey, ykey]

    for i, algo in enumerate(algos):
        path_algo = path.format(algo)
        save_path_algo = None
        df = pd.read_csv(path_algo, header=0, usecols=cols, sep=",", dtype=np.float64, engine="c", low_memory=False)
        x = np.array(df[xkey])
        y = np.array(df[ykey])
        kwargs = {"color": COLORS[i], "xlabel": xlabel,
            "ylabel": ylabel}
        fig_ax = plot_ccurve(x, y, algo=algo, fig_ax=(fig, ax),
            save_path=save_path_algo, **kwargs)

    
    # ax.legend()
    fig.savefig(save_path.format("all"))


    plt.show()