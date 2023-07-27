import multiprocessing as mp
import numpy as np
# from numpy import linalg as la
from scipy import ndimage
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from typing import Callable

from .redux_utils import to_fits, to_npy, numcomps, numworkers

def ADI_npy(cube: np.ndarray, angle_list: np.ndarray,
        collapse: str="median", **kwargs) -> np.ndarray:
    
    collapse_fn = np.median
    if collapse == "mean":
        collapse_fn = np.mean
    elif collapse == "sum":
        collapse_fn = np.sum

    # subtract PSF model from all time frames in this single wavelength channel
    psf = np.median(cube, axis=0)
    cube_res = np.subtract(cube, psf)

    # de-rotate the data
    cube_rot = np.empty_like(cube_res)
    for i in range(len(angle_list)):
        cube_rot[i] = ndimage.rotate(cube_res[i], -1 * angle_list[i], reshape=False)

    # combine de-rotated images in this wavelength channel
    adi = collapse_fn(cube_rot, axis=0)

    return adi

# combine each wavelength channel
def ASDI_npy(cube: np.ndarray, angle_list: np.ndarray,
        collapse_all: str="median", redux_fn: Callable=ADI_npy,
        use_mp: bool=False, **kwargs) -> np.ndarray:
    
    # mp "daemonic" processes cannot spawn further processes
    nproc = kwargs.get("nproc", numworkers) if not use_mp else 1

    combine_fn = np.median
    if collapse_all == "mean":
        combine_fn = np.mean
    elif collapse_all == "sum":
        combine_fn = np.sum

    nchnls = cube.shape[0]
    assert cube.shape[1] == angle_list.shape[0]

    # calculate ADI image for each wavelength channel
    if not use_mp:
        kwargs["full_output"] = False
        chnls = []
        for cube_ch in cube:
            chnls.append(redux_fn(cube=cube_ch, angle_list=angle_list, **kwargs))
        channels = np.array(chnls)

    else:
        # kwargs will really just be positional args... make sure they are good
        input_arg = list(zip(cube, np.repeat([angle_list], nchnls, axis=0),
            np.repeat(kwargs, nchnls, axis=0)))
        with mp.Pool(nproc) as pool:
            channels = np.array(pool.starmap(redux_fn, input_arg))

    # combine ADI images across channels
    adi_combined = combine_fn(channels, axis=0)

    return adi_combined

def PCA_npy(cube: np.ndarray, ncomps: int):
    mtx = cube.reshape(cube.shape[0], -1) #should whiten/center data before pca
    # u, s, vh = la.svd(mtx)  #suggest if m >> n, compute la.qr factorization first
    u, s, vh = svds(mtx, k=ncomps)
    u, s, vh = randomized_svd(mtx, n_components=ncomps)
    trsvd = TruncatedSVD(n_components=ncomps)
    res1 = trsvd.transform(mtx)
    res2 = trsvd.fit_transform(mtx)
    s = trsvd.singular_values_
    vh = trsvd.components_
    u = trsvd(mtx).dot(np.linalg.inv(np.diag(trsvd.singular_values_)))

    return (u, s, vh)


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

    pp_cube = ASDI_npy(channelnums, data_paths, combine_fn=np.mean,
            channel_combine_fn=np.mean)
    
    to_fits(pp_cube, outcombined_path)