import numpy as np
import redux_utils

if __name__ == "__main__":

    # in the future these parameters will change upon running script

    # --- DATA INFO --- #
    # datafilenum = 52
    xdim = 63
    ydim = 63
    firstchannelnum = 40
    lastchannelnum = 80
    channelnums = list(range(firstchannelnum, lastchannelnum))
    nchnls = len(channelnums)
    # --- DATA INFO --- #

    # --- PATHS --- #
    data_dir = "./data"
    out_dir = "./out"
    data_path = "./data/005_center_multishift/wl_channel_%05i.fits"
    data_paths = [data_path] * nchnls
    # --- PATHS --- #

    # --- COMBINE INFO --- #
    combine_fn = np.median
    channel_combine_fn = np.median
    collapse_channel = "median"
    # --- COMBINE INFO --- #

    # --- RUN INFO --- #
    mode = 2
    algo = "PCA"
    # redux_utils.numworkers = 8
    # --- RUN INFO --- #
    
    if mode == 0:       # Classical ADI
        import redux_utils

        # outchannelpath = "./out/cADI_%05i_mean.fits"
        outchannel_path = None
        outchannel_paths = [outchannel_path] * nchnls
        outcombined_path = "./out/cADI_median_%05i_%05i.fits"%(firstchannelnum, lastchannelnum)

        redux_utils.combine_channels(channelnums, data_paths,
                combine_fn=combine_fn, channel_combine_fn=channel_combine_fn,
                out_path=outcombined_path, outchannel_paths=outchannel_paths)
        
    elif mode == 1:     # VIP ADI (LOCI, PCA)
        import redux_vip

        # outchannelpath = "./out/vipADI_%05i_median.fits"
        outchannel_path = None
        outchannel_paths = [outchannel_path] * nchnls
        outcombined_path = "./out/vipADI_temp_median_%05i_%05i.fits"%(firstchannelnum, lastchannelnum)

        if algo == "LOCI":
            redux_vip.combine_channels_vip(channelnums, data_paths=data_paths,
                    combine_fn=combine_fn, collapse_channel=collapse_channel,
                    out_path=outcombined_path, outchannel_paths=outchannel_paths)
        elif algo == "PCA":
            redux_vip.PCA_vip(channelnums)
        else:
            print("Algorithm '%s' not supported for VIP analysis"%algo)
        
    elif mode == 2:     # PynPoint ADI
        import redux_pyn

        data_path = data_dir + "/" + "005_center_multishift/wl_channel_%05i.fits"
        data_paths = [data_path%(channelnum) for channelnum in channelnums]
        outcombined_path = "./out/pyn_PCA%03i_%05i_%05i_PREPPED.fits"%(redux_utils.numcomps, firstchannelnum, lastchannelnum)

        redux_pyn.reduce_channel_pyn(data_dir, data_paths, out_dir=out_dir,
                out_path=outcombined_path)
