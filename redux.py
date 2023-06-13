from astropy.io import ascii
from astropy.io import fits
import numpy as np
from scipy import ndimage

# --- PATHS --- #
datadirpath = "./data/005_center_multishift/images/"
datafilenum = 52
datafilenums = np.arange(40, 80)
datafilename = "wl_channel_%05i.fits"%(datafilenum)
datafilepath = datadirpath + datafilename

outdirname = "out/"
outmedpsffilename = "data_%05i_medpsf.fits"%(datafilenum)
outrotfilename = "data_%05i_rot.fits"%(datafilenum)
outsumfilename = "data_%05i_sum.fits"%(datafilenum)
outmeanfilename = "data_%05i_mean.fits"%(datafilenum)

outmedpsffilepath = outdirname + outmedpsffilename
outrotfilepath = outdirname + outrotfilename
outsumfilepath = outdirname + outsumfilename
outmeanfilepath = outdirname + outmeanfilename

anglesfilepath = "data/parangs_bads_removed.txt"

def savedata(data, path, overwrite=True):
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=data))
    hdul.writeto(path, overwrite=overwrite)
# --- PATHS --- #

# get list of angles for de-rotation
anglestable = ascii.read(anglesfilepath, format="no_header", data_start=0)
angles = anglestable['col1'].data

# load data cube
with fits.open(datafilepath) as datahdu:
    data = datahdu[0].data

# subtract median PSF from all frames in this wavelength channel
medpsf = np.median(data, axis=0)
datasub = np.subtract(data, medpsf)

assert len(angles) == len(datasub)
datarot = np.empty_like(datasub)
for i in range(len(angles)):
    datarot[i] = ndimage.rotate(datasub[i], -1 * angles[i], reshape=False)

# combine de-rotated images in this wavelength channel
datamean = np.mean(datarot, axis=0)
