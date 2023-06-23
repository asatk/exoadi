import multiprocessing as mp
import numpy as np
from typing import Callable

# import vip_hci as vip
# from hciplot import plot_frames, plot_cubes
# from vip_hci.config import VLT_NACO
# from vip_hci.fm import normalize_psf
from vip_hci.psfsub import median_sub, pca
from vip_hci.fits import open_fits, write_fits, info_fits
# from vip_hci.metrics import significance, snr, snrmap
# from vip_hci.var import fit_2dgaussian, frame_center

import redux_utils

def reduce_channel_vip(channelnum: int, datapath: str, collapse: str="median", outpath: str = None):
    
    data = open_fits(datapath%(channelnum))
    dataselected = data[::redux_utils.everynthframe]
    anglesselected = redux_utils.angles[::redux_utils.everynthframe]
    adi = median_sub(dataselected, anglesselected, collapse=collapse, imlib="vip-fft", interpolation=None)

    if outpath is not None:
        redux_utils.savedata(adi, outpath%(channelnum))

    return adi

def combine_channels_vip(channelnums: list[int], datapaths: list[str], combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean, collapse_channel: str="median", outpath: str=None, outchannelpaths: list[str]=None):
    
    nchnls = len(channelnums)
    assert nchnls == len(datapaths)
    
    with mp.Pool(redux_utils.numworkers) as pool:
        adicube = np.array(pool.starmap(
            reduce_channel_vip,
            zip(channelnums, datapaths, [collapse_channel] * nchnls, outchannelpaths)))

    adi_combined = combine_fn(adicube, axis=0)

    redux_utils.savedata(adi_combined, outpath)

if __name__ == "__main__":

    # --- DATA INFO --- #
    firstchannelnum = 40
    lastchannelnum = 80
    channelnums = list(range(firstchannelnum, lastchannelnum))
    nchnls = len(channelnums)
    # --- DATA INFO --- #

    # --- PATHS --- #
    datapath = "./data/005_center_multishift/wl_channel_%05i.fits"
    datapaths = [datapath] * nchnls

    outchannelpath = "./out/vipADI_%05i_median.fits"
    # outchannelpath = None
    outchannelpaths = [outchannelpath] * nchnls
    outcombinedpath = "./out/vipADI_%05i_%05i_median.fits"%(firstchannelnum, lastchannelnum)
    # --- PATHS --- #

    combine_channels_vip(channelnums, datapaths=datapaths, combine_fn=np.mean, collapse_channel="median", outpath=outcombinedpath, outchannelpaths=outchannelpaths)