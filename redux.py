import numpy as np

import analysis
import redux_utils

if __name__ == "__main__":

    # in the future these parameters will change upon running script

    # --- DATA INFO --- #
    # datafilenum = 52
    xdim = 63
    ydim = 63
    firstchannelnum = 45
    lastchannelnum = 74
    channelnums = list(range(firstchannelnum, lastchannelnum + 1))
    nchnls = len(channelnums)
    angles = redux_utils.angles
    angles_skipped = angles[::redux_utils.everynthframe]
    wavelengths = redux_utils.wavelengths[firstchannelnum:lastchannelnum + 1]
    # --- DATA INFO --- #

    # --- PATHS --- #
    data_dir = "./data"
    out_dir = "./out"
    data_path = "./data/005_center_multishift/wl_channel_%05i.fits"
    data_paths = [data_path%i for i in channelnums]
    # --- PATHS --- #

    # --- COMBINE INFO --- #
    combine_fn = np.median
    channel_combine_fn = np.median
    collapse_channel = "median"
    # --- COMBINE INFO --- #

    # --- RUN INFO --- #
    mode = 1
    algo = "ASDI"
    # redux_utils.numworkers = 8
    # --- RUN INFO --- #

    # --- DATA --- #
    cube = redux_utils.loadall([data_path%num for num in channelnums])
    cube_skipped = cube[:, ::redux_utils.everynthframe]
    
    if mode == 0:       # Classical ADI
        import redux_npy

        # outchannel_path = None
        # outchannel_paths = [outchannel_path] * nchnls
        outchannel_path = "./out/cADI_%05i_mean.fits"
        outchannel_paths = [outchannel_path%i for i in channelnums]
        outcombined_path = "./out/cADI_median_%05i_%05i_%02i.fits"%(firstchannelnum, lastchannelnum, redux_utils.everynthframe)

        redux_npy.combine_ADI_npy(cube_skipped, angles_skipped, combine_fn=combine_fn,
            channel_combine_fn=channel_combine_fn, save_fn=redux_utils.to_fits,
            out_path=outcombined_path, outchannel_paths=outchannel_paths)
        
    elif mode == 1:     # VIP ADI
        import redux_vip
        
        # outchannel_path = None
        # outchannel_paths = [outchannel_path] * nchnls
        outchannel_path = "./out/vipADI_%05i_median.fits"
        outchannel_paths = [outchannel_path%i for i in channelnums]
        outcombined_path = "./out/vip%s_median_%05i_%05i_%02i.fits"%(algo, firstchannelnum, lastchannelnum, redux_utils.everynthframe)

        annular_kwargs = {}

        if algo == "ASDI":
            med_asdi, fwhm = redux_vip.ASDI_vip(
                cube=cube, wavelengths=wavelengths, angles=angles,
                out_path=outcombined_path, do_opt=False, kwargs=annular_kwargs)
            
            analysis.calc_stats(med_asdi, fwhm)

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
        # elif algo == "PCA":
        #     redux_vip.PCA_vip(channelnums)
        # elif algo == "TRAP":
        #     pass
        else:
            print("Algorithm '%s' not supported for VIP analysis"%algo)
        
    elif mode == 2:     # PynPoint ADI
        import redux_pyn

        data_path = data_dir + "/" + "005_center_multishift/wl_channel_%05i.fits"
        data_paths = [data_path%(channelnum) for channelnum in channelnums]
        outcombined_path = "./out/pyn_PCA%03i_%05i_%05i.fits"%(redux_utils.numcomps, firstchannelnum, lastchannelnum)

        redux_pyn.reduce_channel_pyn(data_dir, data_paths, out_dir=out_dir,
                out_path=outcombined_path)
