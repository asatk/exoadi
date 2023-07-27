from matplotlib import pyplot as plt
# import multiprocessing as mp
import numpy as np
# import pandas as pd

from mypkg.redux import redux_utils as rxu, redux_vip as rxv
from mypkg.redux.redux_npy import ADI_npy, PCA_npy

from hciplot import plot_frames, plot_cubes
# from vip_hci.var import frame_center, mask_circle
from vip_hci.metrics import completeness_curve, contrast_curve, detection
# from vip_hci.metrics import inverse_stim_map, significance, snr, snrmap, stim_map, throughput
from vip_hci.fm import cube_planet_free
# from vip_hci.fm import firstguess, normalize_psf
# from vip_hci.preproc import find_scal_vector, frame_rescaling
from vip_hci.psfsub import frame_diff, median_sub, pca, pca_annular, xloci


if __name__ == "__main__":

    # --- RUN INFO
    lib = "vip"
    algo = "ASDI"
    sub_type = "ADI"
    first_chnl = 45
    last_chnl = 74
    nframes = 2202
    nskip_frames = 20
    channels = list(range(first_chnl, last_chnl + 1))
    frames = range(0, nframes, nskip_frames)
    ncomp = 10

    # --- DATA INFO
    name_kwargs = {"lib": lib, "algo": algo, "sub_type": sub_type,
                   "first_chnl": first_chnl, "last_chnl": last_chnl,
                   "ncomp": ncomp, "nskip_frames": nskip_frames}

    data_path = "./data/005_center_multishift/wl_channel_%05i.fits"
    data_paths = [data_path%i for i in channels]
    name = rxu.make_name(**name_kwargs)
    pp_path = "out/%s.fits"%name
    pf_path = "out/PF_%s.fits"%name
    wavelengths_path = "data/channel_wavelengths.txt"
    angles_path = "data/parangs_bads_removed.txt"


    # --- INIT INFO
    mask_rad = 10
    do_opt = False


    cubes, wavelengths, angles = rxu.init(data_paths, wavelengths_path,
        angles_path, channels=channels, frames=frames)
    fwhm, psfn, opt_scal, opt_flux = rxv.prep(cubes=cubes,
        wavelengths=wavelengths, mask_rad=mask_rad, do_opt=do_opt)
    

    nchnls = len(wavelengths)
    combine_fn = np.median
    full_output = True

    nbranch = 5
    scaling = "temp-standard"
    nproc = rxu.numworkers
    pxscale = 0.035
    simplex_data = ([21.403], [154.730], [45.632])
    planet_parameter = np.transpose(simplex_data)
    pl_loc = (11.65, 40.14)

    starphot = simplex_data[2]
    nbranch = 5
    theta = 0
    inner_rad = int(max(1., mask_rad / fwhm))
    fc_rad_sep = 3
    noise_sep = 1
    student = True
    smooth = True
    interp_order = 2
    debug = True
    verbose = True
    plot = True
    full_output = True
    imlib = "vip-fft"
    interpolation = "lancsoz4"

    asize = fwhm
    delta_rot = (0.1, 1.0)
    delta_sep = 1.0
    nframes = "auto"
    collapse = "median"
    cubes_used = cubes  # for redux
    # cubes_used = cubes_pf # for ccurve

    planet_parameters = np.repeat([np.array(simplex_data)], nchnls, axis=2)
    cubes_pf = cube_planet_free(planet_parameter=planet_parameters, cube=cubes,
        angs=angles, psfn=psfn, imlib=imlib, interpolation=interpolation,
        transmission=None)
    
    # --- ARGS
    kwargs = {"collapse": "median", "nproc": nproc}
    pca_kwargs = {"collapse_ifs": "mean"}
    ann_kwargs = {"asize": asize, "delta_rot": delta_rot,
                  "delta_sep": delta_sep}
    rot_options = {"imlib": imlib, "interpolation": interpolation,
                   "interp_zeros": True, "mask_val": 0}

    # - NPY
    args_npy_adi = {**kwargs}
    args_npy_pca = {}

    # - MISC
    args_fd = {"mertric": "l1", "dist_threshold": 90, "delta_rot": delta_rot[0],
            "radius_int": mask_rad, "asize": asize, **kwargs, **rot_options}
    args_loci = {**kwargs, **rot_options}

    # - ASDI
    args_asdi = {"scale_list": opt_scal, "flux_sc_list": opt_flux,
                "radius_int": mask_rad, **kwargs}
    args_adi = {"collapse": collapse, **kwargs}
    args_sdi = {"scale_list": opt_scal, "sdi_only": True, **kwargs}
    args_asdi_ann = {"scale_list": opt_scal, "flux_sc_list": opt_flux,
                "mode": "annular", "radius_int": mask_rad, **kwargs, **ann_kwargs}

    # - PCA
    args_sng = {"scale_list": opt_scal, "ncomp": ncomp, "adimsdi":"single",
                "crop_ifs": False, "mask_center_px": mask_rad, "scaling": scaling,
                **kwargs, **pca_kwargs}
    args_dbl = {"scale_list": opt_scal, "ncomp": (ncomp, ncomp), "adimsdi":"double",
                "crop_ifs": False, "mask_center_px": mask_rad, "scaling": scaling,
                **kwargs, **pca_kwargs}
    args_ann = {"scale_list": opt_scal, "ncomp": (ncomp, ncomp),"radius_int": mask_rad,
                **kwargs, **pca_kwargs, **ann_kwargs}


    algo_d = {"asdi": median_sub, "adi": median_sub, "sdi": median_sub, "asdi_ann": median_sub,
            "sng": pca, "dbl": pca, "ann": pca_annular,
            "npy_adi": ADI_npy, "npy_pca": PCA_npy,
            "fd": frame_diff, "loci": xloci}
    args_d = {"asdi": args_asdi, "adi": args_adi, "sdi": args_sdi, "asdi_ann": args_asdi_ann,
            "sng": args_sng, "dbl": args_dbl, "ann": args_ann,
            "npy_adi": args_npy_adi, "npy_pca": args_npy_pca,
            "fd": args_fd, "loci": args_loci}