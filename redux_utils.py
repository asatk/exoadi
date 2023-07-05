from astropy.io import ascii
from astropy.io import fits
import multiprocessing as mp
import numpy as np
from scipy import ndimage
from typing import Callable

numworkers = 8      # number of worker threads/processes
numcomps = 3        # number of PCA components
everynthframe = 50   # number of frames 'n' selected from data cube

# get list of angles for de-rotation
# >>>> figure out why angles are not the double precision as in the file
angles_path = "data/parangs_bads_removed.txt"
anglestable = ascii.read(angles_path, format="no_header", data_start=0)
angles = anglestable['col1'].data

# save array to specified path
def savedata(data: np.ndarray, path: str, overwrite: bool=True) -> None:
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=data))
    hdul.writeto(path, overwrite=overwrite)

def reduce_channel(channelnum: int, data_path: str, 
        combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean,
        out_path: str=None) -> np.ndarray:

    # load data cube
    with fits.open(data_path%(channelnum)) as datahdu:
        data = datahdu[0].data
        assert len(angles) == len(data)

    anglesselected = angles[::everynthframe]
    dataselected = data[::everynthframe]

    # subtract median PSF from all frames in this wavelength channel
    medpsf = np.median(dataselected, axis=0)
    datasub = np.subtract(dataselected, medpsf)

    # de-rotate the data
    datarot = np.empty_like(datasub)
    for i in range(len(anglesselected)):
        datarot[i] = ndimage.rotate(datasub[i], -1 * anglesselected[i], reshape=False)

    # combine de-rotated images in this wavelength channel
    datacombined = combine_fn(datarot, axis=0)

    if out_path is not None:
        savedata(datacombined, out_path%(channelnum))

    return datacombined

# combine each wavelength channel
def combine_channels(channelnums: list[int], data_paths: list[str],
        combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean,
        channel_combine_fn: Callable[[np.ndarray], np.ndarray]=np.mean,
        out_path: str=None, outchannel_paths: list[str]=None) -> np.ndarray:

    nchnls = len(channelnums)
    assert nchnls == len(data_paths)

    if outchannel_paths is not None:
        assert nchnls == len(outchannel_paths)
    else:
        outchannel_paths = list[None] * nchnls

    # calculate ADI image for each wavelength channel
    with mp.Pool(numworkers) as pool:
        channels = np.array(pool.starmap(
            reduce_channel,
            zip(channelnums, data_paths, [channel_combine_fn] * nchnls, outchannel_paths)))

    # combine ADI images across channels
    redux = combine_fn(channels, axis=0)
    
    if out_path is not None:
        savedata(redux, out_path)

    return redux

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

    # outchannelpath = "./out/cADI_%05i_mean.fits"
    outchannel_path = None
    outchannel_paths = [outchannel_path] * nchnls
    outcombined_path = "./out/cADI_mean_%05i_%05i_every%02i.fits"%(firstchannelnum, lastchannelnum, everynthframe)
    # --- PATHS --- #

    combine_channels(channelnums, data_paths, combine_fn=np.mean, channel_combine_fn=np.mean, out_path=outcombined_path, outchannel_paths=outchannel_paths)