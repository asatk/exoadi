import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# from mypkg.redux import redux_utils

COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

def plot_ccurve(x: np.ndarray, y: np.ndarray, algo: str,
        save_path: str=None, fig_ax: tuple[plt.Figure, plt.Axes]=None,
        logy: bool=False, **kwargs):
    
    color = kwargs.get("color", None)
    xlabel = kwargs.get("xlabel", None)
    ylabel = kwargs.get("ylabel", None)
    xlims = kwargs.get("xlims", (None, None))
    ylims = kwargs.get("ylims", (None, None))
    title = kwargs.get("title", "Constrast Curve")

    
    xstart = x[0]
    xend = x[-1]

    ystart = y[0]
    yend = y[-1]
    
    # xy = [xstart, ystart]
    xy = [xend, yend]
    # xytext = [xstart*(1-0.075), ystart*(1+0.1)] if logy else [xstart-0.055, ystart-0.05]
    xytext = [xend+0.005, yend*(1+0.1) if logy else yend+0.05]
    

    fig, ax = plt.subplots(figsize=(8, 6)) if fig_ax is None else fig_ax

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.plot(x, y, label=algo, color=color, alpha=0.75)
    if logy:
        ax.set_yscale("log")
    ax.annotate(text=algo, xy=xy, xytext=xytext, color=color)
    ax.plot(x, y, label=algo, color=color, alpha=0.75)
    # ax.set_xlim(xlims[0], xlims[1])
    # ax.set_ylim(ylims[0], ylims[0])
    # ax.legend()
    # if not y_in_unit_intv:
    #     ax.hlines([0.0, 1.0], xmin=0.0, xmax=xmax, linestyles="dashed", colors="gray", alpha=0.75)
    
    if save_path is not None:
        fig.savefig(save_path)
    
    return fig, ax

if __name__ == "__main__":

    slice_after = 4
    logy = False
    use_arcsec = True
    use_mag = True
    scale = "arcsec" if use_arcsec else "px"
    # path = "out/datafr_vipPCA003-{:s}_45-74_skip20.csv"
    save_path = "out/cc_vipPCA003-{:s}_45-74_skip20.png"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ncomp = 4
    algo_names = [
                "sng",
                "dbl",
                "ann",
                "asdi",
                "sdi",
                "adi",
                "adi_ann",
                "npy_adi"
    ]
    # algo_names = ["sng", "dbl", "ann",
    #               "asdi", "sdi", "adi", "adi_ann",
    #               "npy_adi", "fd", "loci"]
    
    df_path_fmt = "out/df_{:s}_45-74_20.csv"
    df_paths = []
    for algo_name in algo_names:
        if algo_name in ["sng", "dbl", "ann"]:
            df_path = df_path_fmt.format(algo_name + f"{ncomp:03}")
        else:
            df_path = df_path_fmt.format(algo_name)
        df_paths.append(df_path)


    xkey = "distance_arcsec" if use_arcsec else "distance"
    xlabel = f"Distance ({scale})"
    # ykey = "throughput"
    # ykey = "sensitivity_gaussian"
    ykey = "sensitivity_student"
    # ylabel = "Throughput"
    # ylabel = "5$\sigma$ Constrast"
    ylabel = "$\Delta$mag"

    cols = [xkey, ykey]

    xlims = (None, None)
    ylims = (None, None)

    for i, path in enumerate(df_paths):
        algo_name = algo_names[i]
        save_path_algo = None
        df = pd.read_csv(path, header=0, usecols=cols, sep=",", dtype=np.float64, engine="c", low_memory=False)
        x = np.array(df[xkey][slice_after:])
        y = np.array(df[ykey][slice_after:])
        if use_mag:
            y = -2.5 * np.log10(y)
            print(y)
        kwargs = {"color": COLORS[i], "xlabel": xlabel,
            "ylabel": ylabel, "xlims": xlims, "ylims": ylims}
        fig_ax = plot_ccurve(x, y, algo=algo_name, fig_ax=(fig, ax),
            save_path=save_path_algo, logy=logy, **kwargs)

    
    # ax.legend()
    fig.savefig(save_path.format("all"))


    plt.show()