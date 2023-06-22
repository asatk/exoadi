from astropy.io import ascii
from astropy.io import fits
import multiprocessing as mp
import numpy as np
from scipy import ndimage
from typing import Callable

numworkers = 8
numcomps = 20


# get list of angles for de-rotation
# >>>> figure out why angles are not the double precision as in the file
anglespath = "data/parangs_bads_removed.txt"
anglestable = ascii.read(anglespath, format="no_header", data_start=0)
angles = anglestable['col1'].data

# save array to specified path
def savedata(data: np.ndarray, path: str, overwrite: bool=True) -> None:
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=data))
    hdul.writeto(path, overwrite=overwrite)

def reduce_channel(channelnum: int, datapath: str, combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean, outpath: str=None) -> np.ndarray:

    # load data cube
    with fits.open(datapath%(channelnum)) as datahdu:
        data = datahdu[0].data
        assert len(angles) == len(data)

    # subtract median PSF from all frames in this wavelength channel
    medpsf = np.median(data, axis=0)
    datasub = np.subtract(data, medpsf)

    # de-rotate the data
    datarot = np.empty_like(datasub)
    for i in range(len(angles)):
        datarot[i] = ndimage.rotate(datasub[i], -1 * angles[i], reshape=False)

    # combine de-rotated images in this wavelength channel
    datacombined = combine_fn(datarot, axis=0)

    if outpath is not None:
        savedata(datacombined, outpath%(channelnum))

    return datacombined

# combine each wavelength channel
def combine_channels(channelnums: list[int], datapaths: list[str], combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean, channel_combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean, outpath: str=None, outchannelpaths: list[str]=None) -> np.ndarray:

    nchnls = len(channelnums)
    assert nchnls == len(datapaths)

    if outchannelpaths is not None:
        assert nchnls == len(outchannelpaths)
    else:
        outchannelpaths = list[None] * nchnls

    # calculate ADI image for each wavelength channel
    with mp.Pool(numworkers) as pool:
        channels = np.array(pool.starmap(
            reduce_channel,
            zip(channelnums, datapaths, [channel_combine_fn] * nchnls, outchannelpaths)))

    # combine ADI images across channels
    redux = combine_fn(channels, axis=0)
    
    if outpath is not None:
        savedata(redux, outpath)

    return redux

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

    outchannelpath = "./out/cADI_%05i_mean.fits"
    # outchannelpath = None
    outchannelpaths = [outchannelpath] * nchnls
    outcombinedpath = "./out/cADI_mean_%05i_%05i.fits"%(firstchannelnum, lastchannelnum)
    # --- PATHS --- #

    combine_channels(channelnums, datapaths, combine_fn=np.mean, channel_combine_fn=np.mean, outpath=outcombinedpath, outchannelpaths=outchannelpaths)