import numpy as np

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
    datapath = "./data/005_center_multishift/wl_channel_%05i.fits"
    datapaths = [datapath] * nchnls
    # --- PATHS --- #

    # --- COMBINE INFO --- #
    combine_fn = np.median
    channel_combine_fn = np.median
    collapse_channel = "median"
    # --- COMBINE INFO --- #

    # --- RUN INFO --- #
    mode = 1
    # redux_utils.numworkers = 8
    # --- RUN INFO --- #
    
    if mode == 0:       # Classical ADI
        import redux_utils

        # outchannelpath = "./out/cADI_%05i_mean.fits"
        outchannelpath = None
        outchannelpaths = [outchannelpath] * nchnls
        outcombinedpath = "./out/cADI_median_%05i_%05i.fits"%(firstchannelnum, lastchannelnum)

        redux_utils.combine_channels(channelnums, datapaths,
                combine_fn=combine_fn, channel_combine_fn=channel_combine_fn,
                outpath=outcombinedpath, outchannelpaths=outchannelpaths)
        
    elif mode == 1:     # VIP ADI (LOCI)
        import redux_vip

        # outchannelpath = "./out/vipADI_%05i_median.fits"
        outchannelpath = None
        outchannelpaths = [outchannelpath] * nchnls
        outcombinedpath = "./out/vipADI_temp_median_%05i_%05i.fits"%(firstchannelnum, lastchannelnum)

        redux_vip.combine_channels_vip(channelnums, datapaths=datapaths,
                combine_fn=combine_fn, collapse_channel=collapse_channel,
                outpath=outcombinedpath, outchannelpaths=outchannelpaths)
        
    elif mode == 2:     # PynPoint ADI
        import redux_pyn

        outchannelpath = "./out/pynADI_%05i_median.fits"
        # outchannelpath = None
        outchannelpaths = [outchannelpath] * nchnls
        outcombinedpath = "./out/pynADI_%05i_%05i_median.fits"%(firstchannelnum, lastchannelnum)
