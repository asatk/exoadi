from astropy.io import ascii
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np

import vip_hci as vip
from hciplot import plot_frames, plot_cubes
from vip_hci.config import VLT_NACO
from vip_hci.fm import normalize_psf
from vip_hci.psfsub import median_sub, pca
from vip_hci.fits import open_fits, write_fits, info_fits
from vip_hci.metrics import significance, snr, snrmap
from vip_hci.var import fit_2dgaussian, frame_center

import redux_utils

numworkers = 8
firstchannelnum = 40
lastchannelnum = 80
datafilenums = list(range(firstchannelnum, lastchannelnum))
nchnls = len(datafilenums)

outchannelfilepath = "./out/vipADI_%05i_median.fits"
outchannelfilepaths = [outchannelfilepath] * nchnls
outallfilepath = "./out/vipADI_%05i_%05i_median.fits"

# get list of angles for de-rotation
anglesfilepath = "data/parangs_bads_removed.txt"
anglestable = ascii.read(anglesfilepath, format="no_header", data_start=0)
angles = anglestable['col1'].data

datadirpath = "./data/005_center_multishift/"
datafilename = "wl_channel_%05i.fits"
datapath = datadirpath + datafilename
datafilepaths = [datapath] * nchnls


def reduce_channel_vip(datanum: int, datafilepath: str, outfilepath: str = None):
    
    data = open_fits(datafilepath%(datanum))
    adi = median_sub(data, angles, imlib='vip-fft', interpolation=None)

    if outfilepath is not None:
        redux_utils.savedata(adi, outfilepath%(datanum))

    return adi


with mp.Pool(numworkers) as pool:
    adicube = np.array(pool.starmap(
        reduce_channel_vip,
        zip(datafilenums, datafilepaths, outchannelfilepaths)))

adi_combined = np.median(adicube, axis=0)

redux_utils.savedata(adi_combined, outallfilepath%(firstchannelnum, lastchannelnum))