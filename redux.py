from astropy.io import ascii
from astropy.io import fits
import multiprocessing as mp
import numpy as np
from scipy import ndimage

# --- DATA INFO --- #
# datafilenum = 52
xdim = 63
ydim = 63
firstdatafilenum = 40
lastdatafilenum = 80
datafilenums = range(firstdatafilenum, lastdatafilenum)
# --- DATA INFO --- #

# --- PATHS --- #
datadirpath = "./data/005_center_multishift/"
datafilename = "wl_channel_%05i.fits"
datafilepath = datadirpath + datafilename

outdirname = "out/"
outmedpsffilename = "data_%05i_medpsf.fits"
outrotfilename = "data_%05i_rot.fits"
outsumfilename = "data_%05i_sum.fits"
outmeanfilename = "data_%05i_mean.fits"
outmeanallfilename = "data_mean_%05i_%05i.fits"%(firstdatafilenum, lastdatafilenum)

outmedpsffilepath = outdirname + outmedpsffilename
outrotfilepath = outdirname + outrotfilename
outsumfilepath = outdirname + outsumfilename
outmeanfilepath = outdirname + outmeanfilename
outmeanallfilenpath = outdirname + outmeanallfilename
# --- PATHS --- #

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

def reduce_channel(datafilenum: int) -> np.ndarray:

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
    datamean = np.mean(datarot, axis=0)
    # print("completed channel %i"%(datafilenum))
    # savedata(datamean, outmeanfilepath%(datafilenum))
    return datamean

# calculate mean ADI image for each wavelength channel
with mp.Pool(8) as pool:
    means = np.array(pool.map(reduce_channel, datafilenums))

# combine each wavelength channel
meanall = np.mean(means, axis=0)
savedata(meanall, outmeanallfilenpath)
