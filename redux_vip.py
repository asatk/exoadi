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

def reduce_channel_vip(channelnum: int, data_path: str, collapse: str="median", out_path: str = None):
    
    data = open_fits(data_path%channelnum)
    dataselected = data[::redux_utils.everynthframe]
    anglesselected = redux_utils.angles[::redux_utils.everynthframe]
    adi = median_sub(dataselected, anglesselected, collapse=collapse, imlib="vip-fft", interpolation="biquintic")

    if out_path is not None:
        redux_utils.savedata(adi, out_path%channelnum)

    return adi

def combine_channels_vip(channelnums: list[int], data_paths: list[str], combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean, collapse_channel: str="median", out_path: str=None, outchannel_paths: list[str]=None):
    
    nchnls = len(channelnums)
    assert nchnls == len(data_paths)
    
    with mp.Pool(redux_utils.numworkers) as pool:
        adicube = np.array(pool.starmap(
            reduce_channel_vip,
            zip(channelnums, data_paths, [collapse_channel] * nchnls, outchannel_paths)))

    adi_combined = combine_fn(adicube, axis=0)

    redux_utils.savedata(adi_combined, out_path)

def PCA_vip(channelnums: list[int]) -> None:
    pass

if __name__ == "__main__":

    # --- DATA INFO --- #
    firstchannelnum = 40
    lastchannelnum = 80
    channelnums = list(range(firstchannelnum, lastchannelnum))
    nchnls = len(channelnums)
    # --- DATA INFO --- #

    # --- PATHS --- #
    data_path = "./data/005_center_multishift/wl_channel_%05i.fits"
    data_paths = [data_path] * nchnls

    outchannel_path = "./out/vipADI_%05i_median.fits"
    # outchannelpath = None
    outchannel_paths = [outchannel_path] * nchnls
    outcombined_path = "./out/vipADI_%05i_%05i_median.fits"%(firstchannelnum, lastchannelnum)
    # --- PATHS --- #

    combine_channels_vip(channelnums, data_paths=data_paths, combine_fn=np.mean, collapse_channel="median", out_path=outcombined_path, outchannel_paths=outchannel_paths)