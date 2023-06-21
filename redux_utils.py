from astropy.io import ascii
from astropy.io import fits
import multiprocessing as mp
import numpy as np
from scipy import ndimage
from typing import Callable

numworkers = 8

# get list of angles for de-rotation
anglesfilepath = "data/parangs_bads_removed.txt"
anglestable = ascii.read(anglesfilepath, format="no_header", data_start=0)
angles = anglestable['col1'].data

# save array to specified path
def savedata(data: np.ndarray, path: str, overwrite: bool=True) -> None:
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=data))
    hdul.writeto(path, overwrite=overwrite)

def reduce_channel(datafilenum: int, datafilepath: str, combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean, outfilepath: str=None) -> np.ndarray:

    # load data cube
    with fits.open(datafilepath%(datafilenum)) as datahdu:
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

    if outfilepath is not None:
        savedata(datacombined, outfilepath%(datafilenum))

    return datacombined

# combine each wavelength channel
def combine_channels(datafilenums: list[int], datafilepaths: list[str], combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean, channel_combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean, outfilepath: str=None, outchannelfilepaths: list[str]=None) -> np.ndarray:

    nchnls = len(datafilenums)
    assert nchnls == len(datafilepaths)

    if outchannelfilepaths is not None:
        assert nchnls == len(outchannelfilepaths)
    else:
        outchannelfilepaths = list[None] * nchnls

    # calculate ADI image for each wavelength channel
    with mp.Pool(numworkers) as pool:
        channels = np.array(pool.starmap(
            reduce_channel,
            zip(datafilenums, datafilepaths, [channel_combine_fn] * nchnls, outchannelfilepaths)))

    # combine ADI images across channels
    redux = combine_fn(channels, axis=0)
    
    if outfilepath is not None:
        savedata(redux, outfilepath)

    return redux