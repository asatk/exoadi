import multiprocessing as mp
import numpy as np
from numpy import linalg as la
from scipy import ndimage
from typing import Callable

from redux_utils import to_fits, to_npy, numcomps, numworkers

def ADI_npy(cube: np.ndarray, angles: np.ndarray,
        combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean,
        save_fn: Callable[[np.ndarray, str, bool], None]=to_fits,
        out_path: str=None) -> np.ndarray:

    # subtract PSF model from all time frames in this single wavelength channel
    psf = np.median(cube, axis=0)
    cube_res = np.subtract(cube, psf)

    # de-rotate the data
    cube_rot = np.empty_like(cube_res)
    for i in range(len(angles)):
        cube_rot[i] = ndimage.rotate(cube_res[i], -1 * angles[i], reshape=False)

    # combine de-rotated images in this wavelength channel
    adi = combine_fn(cube_rot, axis=0)
    if out_path is not None:
        save_fn(adi, out_path)

    return adi

# combine each wavelength channel
def combine_ADI_npy(cube: np.ndarray, angles: np.ndarray,
        combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean,
        channel_combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean,
        save_fn: Callable[[np.ndarray, str, bool], None]=to_fits,
        out_path: str=None, outchannel_paths: list[str]=None) -> np.ndarray:

    nchnls = cube.shape[0]
    assert cube.shape[1] == angles.shape[0]
    if outchannel_paths is not None:
        assert nchnls == len(outchannel_paths)
    else:
        outchannel_paths = list[None] * nchnls

    # calculate ADI image for each wavelength channel
    with mp.Pool(numworkers) as pool:
        channels = np.array(pool.starmap(
            ADI_npy,
            zip(cube, angles, [channel_combine_fn] * nchnls, [save_fn] * nchnls, outchannel_paths)))

    # combine ADI images across channels
    adi_combined = combine_fn(channels, axis=0)
    if out_path is not None:
        save_fn(adi_combined, out_path)

    return adi_combined

# def PCA_npy(cube: np.ndarray, ncomps: int):
#     mtx = cube.reshape(cube.shape[0], -1)
    # s = la.svd(mtx,)  #suggest if m >> n, compute la.qr factorization first


if __name__ == "__main__":

    # --- DATA INFO --- #
    firstchannelnum = 45
    lastchannelnum = 74
    channelnums = list(range(firstchannelnum, lastchannelnum + 1))
    nchnls = len(channelnums)
    # --- DATA INFO --- #

    # --- PATHS --- #
    data_path = "./data/005_center_multishift/wl_channel_%05i.fits"
    data_paths = [data_path%i for i in channelnums]

    # outchannel_path = None
    # outchannel_paths = [outchannel_path] * nchnls
    outchannel_path = "./out/vipADI_%05i_median.fits"
    outchannel_paths = [outchannel_path%i for i in channelnums]
    outcombined_path = "./out/vipADI_%05i_%05i_median.fits"%(firstchannelnum, lastchannelnum)
    # --- PATHS --- #

    combine_ADI_npy(channelnums, data_paths, combine_fn=np.mean,
            channel_combine_fn=np.mean, save_fn=to_fits,
            out_path=outcombined_path, outchannel_paths=outchannel_paths)