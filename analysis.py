from astropy.io import fits
from astropy.modeling import models, fitting
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np

datafilepath = "./out/data_mean_00040_00080.fits"

with fits.open(datafilepath) as datahdu:
    data = datahdu[0].data

xdim = 63
ydim = 63
subdata = data[0:xdim/2, 0:ydim/2]

gausmodel = models.Gaussian2D(amplitude=np.max(subdata), x_mean=xdim/4, y_mean=ydim/4)