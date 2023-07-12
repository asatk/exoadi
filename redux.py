from matplotlib import pyplot as plt
import numpy as np

import analysis
import redux_utils
from redux_vip import prep
from analysis import detect_planet, find_planet


def postproc(cubes, wavelengths, angles, lib: str="vip", algo: str="ASDI",
             sub_type: str=None, full_output: bool=False,
             channel_combine_fn=np.median, combine_fn=np.median,
             annular_kwargs: dict=None) -> np.ndarray:

    if annular_kwargs is None:
        annular_kwargs = {"asize": None, "delta_rot": None, "delta_sep": None,
                          "nframes": None, "n_segments": None}

    if lib == "npy":       # Classical ADI
        import redux_npy

        pp_cubes = redux_npy.combine_ADI_npy(cubes, angles, combine_fn=combine_fn,
            channel_combine_fn=channel_combine_fn)
        
    elif lib == "vip":     # VIP ADI
        import redux_vip

        if algo == "ASDI":
            pp_cubes = redux_vip.ASDI_vip(cubes, wavelengths, angles,
                do_opt=False, full_output=full_output, sub_type=sub_type,
                **annular_kwargs)

        # elif algo == "ANDROMEDA":
        #     pass
        # elif algo == "FMMF":
        #     pass
        # elif algo == "LLSG":
        #     pass
        # elif algo == "LOCI":
        #     pass
        # elif algo == "NMF":
        #     pass
        # elif algo == "PACO":
        #     pass
        elif algo == "PCA":
            pp_cubes = redux_vip.PCA_vip(cubes, wavelengths, angles,
                do_opt=False, full_output=full_output, sub_type=sub_type,
                **annular_kwargs)
        # elif algo == "TRAP":
        #     pass
        else:
            print("Algorithm '%s' not supported for VIP analysis"%algo)
            pp_cubes = None

    return pp_cubes


if __name__ == "__main__":

    # in the future these parameters will change upon running script

    # --- DATA INFO --- #
    xdim = 63
    ydim = 63
    nframes = 2202
    first_chnl = 45
    last_chnl = 74
    channels = list(range(first_chnl, last_chnl + 1))
    nchnls = len(channels)
    frames = range(0, nframes, redux_utils.everynthframe)
    # --- DATA INFO --- #


    # --- RUN INFO --- #
    lib = "vip"
    algo = "PCA"
    sub_type = "single"
    full_output = False
    # --- RUN INFO --- #


    # --- PATHS --- #
    data_dir = "./data"
    out_dir = "./out"
    data_path = "./data/005_center_multishift/wl_channel_%05i.fits"
    data_paths = [data_path%i for i in channels]

    # outchannel_path = None
    # outchannel_paths = [outchannel_path] * nchnls
    # outchannel_path = f"./out/{lib}{algo}-{sub_type}_%03i.fits"
    # outchannel_paths = [outchannel_path%i for i in channelnums]
    outcombined_path = f"./out/{lib}{algo}-{sub_type}_{first_chnl}-{last_chnl}_skip{redux_utils.everynthframe}.fits"
    # --- PATHS --- #


    # --- COMBINE INFO --- #
    combine_fn = np.median
    channel_combine_fn = np.median
    collapse_channel = "median"
    # --- COMBINE INFO --- #


    # --- DATA --- #
    # Load important derotation angle and channel wavelength data
    angles_path = "data/parangs_bads_removed.txt"
    wavelengths_path = "data/channel_wavelengths.txt"
    cubes, wavelengths, angles = redux_utils.init(data_paths, wavelengths_path, angles_path, channels=channels, frames=frames)
    # --- DATA --- #

    pp_data = postproc(cubes, wavelengths, angles, lib=lib, algo=algo,
        sub_type=sub_type, full_output=full_output)
    if full_output:
        pp_frame = pp_data[0]
    else:
        pp_frame = pp_data

    redux_utils.to_fits(pp_frame, outcombined_path, overwrite=True)
