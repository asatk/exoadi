import numpy as np

import redux_utils

# --- DATA INFO --- #
# datafilenum = 52
xdim = 63
ydim = 63
firstdatafilenum = 40
lastdatafilenum = 80
datafilenums = list(range(firstdatafilenum, lastdatafilenum))
nchnls = len(datafilenums)
# --- DATA INFO --- #

# --- PATHS --- #
datadirpath = "./data/005_center_multishift/"
datafilename = "wl_channel_%05i.fits"
datafilepath = datadirpath + datafilename
datafilepaths = [datafilepath] * nchnls

outdirname = "out/"
outchannelfilename = "data_%05i_median.fits"
outallfilename = "data_mean_%05i_%05i.fits"%(firstdatafilenum, lastdatafilenum)

# outchannelfilepath = outdirname + outchannelfilename
outchannelfilepath = None
outchannelfilepaths = [outchannelfilepath] * nchnls
outallfilepath = outdirname + outallfilename
# --- PATHS --- #

redux_utils.combine_channels(datafilenums, datafilepaths, combine_fn=np.mean, channel_combine_fn=np.mean, outfilepath=outallfilepath, outchannelfilepaths=outchannelfilepaths)

