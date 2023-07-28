from matplotlib import pyplot as plt
# import multiprocessing as mp
import numpy as np
# import pandas as pd
from typing import Callable

from mypkg.redux import redux_utils as rxu, redux_vip as rxv
from mypkg.redux.redux_npy import ADI_npy, ASDI_npy, PCA_npy

from astropy.visualization import ZScaleInterval
from hciplot import plot_frames, plot_cubes
# from vip_hci.var import frame_center, mask_circle
from vip_hci.metrics import completeness_curve, contrast_curve, detection
# from vip_hci.metrics import inverse_stim_map, significance, snr, snrmap, stim_map, throughput
from vip_hci.fm import cube_planet_free
# from vip_hci.fm import firstguess, normalize_psf
# from vip_hci.preproc import find_scal_vector, frame_rescaling
from vip_hci.psfsub import frame_diff, median_sub, pca, pca_annular, xloci

lib = "vip"
algo = "ASDI"
sub_type = "ADI"

ncomp = 4
first_chnl = 45
last_chnl = 74
nframes = 2202
nskip_frames = 20
channels = list(range(first_chnl, last_chnl + 1))
frames = range(0, nframes, nskip_frames)

name_kwargs = {"lib": lib, "algo": algo, "sub_type": sub_type,
                "first_chnl": first_chnl, "last_chnl": last_chnl,
                "ncomp": ncomp, "nskip_frames": nskip_frames}

data_path = "./data/005_center_multishift/wl_channel_%05i.fits"
data_paths = [data_path%i for i in channels]
wavelengths_path = "data/channel_wavelengths.txt"
angles_path = "data/parangs_bads_removed.txt"


mask_rad = 8
opt_scal_path = f"out/opt_scal_{first_chnl}-{last_chnl}_{nskip_frames}.npy"
opt_flux_path = f"out/opt_flux_{first_chnl}-{last_chnl}_{nskip_frames}.npy"
do_opt = False
load_opt = True
correct_outliers = True

cubes, wavelengths, angles = rxu.init(data_paths, wavelengths_path,
    angles_path, channels=channels, frames=frames)
fwhm, psfn, opt_scal, opt_flux = rxv.prep(cubes=cubes, wavelengths=wavelengths,
    mask_rad=mask_rad, do_opt=do_opt, correct_outliers=correct_outliers)
if do_opt:  # takes ~7min
    np.save(opt_scal_path, opt_scal)
    np.save(opt_flux_path, opt_flux)
elif load_opt:
    opt_scal = np.load(opt_scal_path)
    opt_flux = np.load(opt_flux_path)

nchnls = len(wavelengths)

scaling = "temp-standard"
nproc = rxu.numworkers
pxscale = 0.035
simplex_data = ([21.403], [154.730], [45.632])
planet_parameter = np.transpose(simplex_data)
pl_loc = (11.65, 40.14)

starphot = simplex_data[2]
nbranch = 5
theta = 0
# inner_rad = int(max(1., mask_rad / fwhm))
inner_rad = 1
fc_rad_sep = 3
noise_sep = 1
student = True
smooth = True
interp_order = 2
debug = True
verbose = True
plot = True
full_output = False
imlib = "vip-fft"
interpolation = "lancsoz4"
metric = "l2"

asize = fwhm
delta_rot = (0.1, 1.0)
delta_rot_scalar = 0.5
delta_sep = (0.1, 1.0)
nframes = "auto"
collapse = "median"
collapse_all = "median"

planet_parameters = np.repeat([np.array(simplex_data)], nchnls, axis=2)
cubes_pf = cube_planet_free(planet_parameter=planet_parameters, cube=cubes, angs=angles,
    psfn=psfn, imlib=imlib, interpolation=interpolation, transmission=None)

cubes_used = cubes_pf # for ccurve

def build_args():

    kwargs = {"collapse": collapse, "nproc": nproc}
    pca_kwargs = {"collapse_ifs": "mean"}
    ann_kwargs = {"asize": asize, "delta_rot": delta_rot, "delta_sep": delta_sep}
    rot_options = {"imlib": imlib, "interpolation": interpolation,
                    "interp_zeros": True, "mask_val": 0}
    args_req = {"cube": cubes_used, "angle_list": angles, "fwhm": fwhm,
                "full_output": full_output, "verbose": verbose}

    # - NPY
    args_npy_adi = {"radius_int": mask_rad, **kwargs}
    args_npy_asdi = {"collapse_all": "median", "use_mp": False, "scale_list": opt_scal}
    args_npy_pca = {}

    # - MISC
    args_fd = {"metric": metric, "dist_threshold": 90, "delta_rot": delta_rot_scalar,
            "radius_int": mask_rad, "asize": asize, **kwargs, **rot_options}
    args_loci = {"metric": metric, **kwargs, **rot_options}

    # - ASDI
    args_asdi = {"scale_list": opt_scal, "flux_sc_list": opt_flux,
                "radius_int": mask_rad, **kwargs}
    args_adi = {"radius_int": mask_rad, **kwargs}
    args_sdi = {"scale_list": opt_scal, "flux_sc_list": opt_flux, "sdi_only": True,
                "radius_int": mask_rad, **rot_options}
    args_adi_ann = {"mode": "annular", "radius_int": mask_rad, "asize": int(fwhm), "delta_rot": delta_rot_scalar, **kwargs}
    # args_asdi_ann = {"scale_list": opt_scal, "flux_sc_list": opt_flux,
    #             "mode": "annular", "radius_int": mask_rad, **kwargs, **ann_kwargs}

    # - PCA
    args_sng = {"scale_list": opt_scal, "ncomp": ncomp, "adimsdi": "single",
                "crop_ifs": False, "mask_center_px": mask_rad, "scaling": scaling,
                "delta_rot": delta_rot_scalar, "source_xy": pl_loc, **kwargs, **pca_kwargs}
    args_dbl = {"scale_list": opt_scal, "ncomp": (ncomp, ncomp), "adimsdi": "double",
                "crop_ifs": False, "mask_center_px": mask_rad, "scaling": scaling,
                "delta_rot": delta_rot_scalar, "source_xy": pl_loc, **kwargs, **pca_kwargs}
    args_dbl_sdi = {"scale_list": opt_scal, "ncomp": (ncomp, None), "adimsdi": "double",
                "crop_ifs": False, "mask_center_px": mask_rad, "scaling": scaling,
                "delta_rot": delta_rot_scalar, "source_xy": pl_loc, **kwargs, **pca_kwargs}
    args_ann = {"scale_list": opt_scal, "ncomp": (ncomp, ncomp),"radius_int": mask_rad,
                **kwargs, **pca_kwargs, **ann_kwargs}


    algo_d = {"asdi": median_sub, "adi": ASDI_npy, "adi_ann": ASDI_npy, "sdi": median_sub,
            "sng": pca, "dbl": pca, "dbl_sdi": pca, "ann": pca_annular,
            "npy_asdi": ASDI_npy, "npy_adi": ASDI_npy, "npy_pca": PCA_npy,
            "fd": ASDI_npy, "loci": ASDI_npy}
    args_d = {"asdi": args_asdi,
            "adi": {"redux_fn": median_sub, **args_npy_asdi, **args_adi},
            "adi_ann": {"redux_fn": median_sub, **args_npy_asdi, **args_adi_ann},
            "sdi": args_sdi,
            "sng": args_sng, "dbl": args_dbl, "dbl_sdi": args_dbl_sdi, "ann": args_ann,
            "npy_asdi": args_npy_asdi, 
            "npy_adi": {"redux_fn": ADI_npy, **args_npy_asdi, **args_npy_adi},
            "npy_pca": args_npy_pca,
            "fd": {"redux_fn": frame_diff, **args_npy_asdi, **args_fd},
            "loci": {"redux_fn": xloci, **args_npy_asdi, **args_loci}}

algo_name = "ann"
algo = algo_d[algo_name]
algo_dict = args_d[algo_name]

# fd - t=5:27, max=6
# loci - t=11:11, max=4
# dbl - t=0:18, max=0.4
# dbl_sdi - t=0:10, max=0.1
# sng - t=1:30, max=0.5
# ann - t=0:26, max=5
# asdi - t=0:06, max=2
# adi - t=0:27, max=6
# adi_ann - t=0:39, max=6
# sdi - t=0:19, max=3
# npy_adi - t=0:02.9, max=6
# npy_adi - t=0:02.9, max=6

# args_req & algo_dict
def reduce(algo: Callable, **args):
    res = algo(**args)
    plot_frames(res)

    zmin, zmax = ZScaleInterval().get_limits(res)
    plot_frames(res, vmin=zmin, vmax=zmax)
    return res

# args_req and args_sng
def opt_ncomp(pl_loc, **args):
    # 1,30,3 - t=2:10
    ncomp_tuple = (1, 30, 1)
    args_pca_search = {**args, "ncomp": ncomp_tuple, "source_xy": pl_loc}
    res = pca(**args_pca_search)
    return res


algo_name = "asdi"
algo = algo_d[algo_name]
algo_dict = args_d[algo_name]
cc_kwargs = {"psf_template": psfn, "algo": algo, "pxscale": pxscale, "starphot": starphot,
             "sigma": 5, "nbranch": nbranch, "theta": theta, "inner_rad": inner_rad,
             "fc_rad_sep": fc_rad_sep, "noise_sep": noise_sep, "student": student,
             "smooth": smooth, "interp_order": interp_order, "debug": debug,
             "plot": plot, **rot_options, **algo_dict, **args_req, "full_output" :True}
cc_res = contrast_curve(**cc_kwargs)



df = cc_res[0] if isinstance(cc_res, tuple) else cc_res
out_path_df = f"out/df_{algo_name}_{first_chnl}-{last_chnl}_{nskip_frames}.csv"
df.to_csv("out/df_ann_45-74_20.csv")

