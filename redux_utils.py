from astropy.io import ascii
from astropy.io import fits
import multiprocessing as mp
import numpy as np
from os.path import isfile
from typing import Callable
from vip_hci.fits import open_fits

numworkers = 8      # number of worker threads/processes
numcomps = 3        # number of PCA components
everynthframe = 20  # number of frames 'n' selected from data cube
chunksize = 20

def init(angles_path: str, wavelengths_path: str, frames: list[int]=..., channels: list[int]=...) -> tuple[np.ndarray, np.ndarray]:
    # get list of angles for de-rotation
    # >>>> figure out why angles are not the double precision as in the file

    angles_table = ascii.read(angles_path, format="no_header", data_start=0)
    angles = angles_table["col1"].data[frames].copy()

    # get list of wavelengths for proper calibration and scaling
    wavelengths_table =  ascii.read(wavelengths_path, format="no_header", data_start=0)
    wavelengths = wavelengths_table["col1"].data[channels].copy()

    return angles, wavelengths

# >>>> add function to just open data instead of doing it w/in redux fn
def loadchannel(path: str, verbose: bool=False) -> np.ndarray:
    return open_fits(path, verbose=verbose)

def loadall(paths: list[str], verbose: bool=False) -> np.ndarray:
    with mp.Pool(numworkers) as pool:
        return np.array(pool.starmap(loadchannel, zip(paths, np.repeat([verbose], len(paths)))))

# save numpy array to specified path as .FITS file
def to_fits(data: np.ndarray, path: str, overwrite: bool=True) -> None:
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=data))
    hdul.writeto(path, overwrite=overwrite)

# save numpy array to specified path as .npy file
def to_npy(data: np.ndarray, path: str, overwrite: bool=True) -> None:
    if isfile(path) and not overwrite:
        np.save(path, data)

def combine(channels: np.ndarray,
        combine_fn: Callable[[np.ndarray], np.ndarray]=np.median,
        out_path: str=None) -> np.ndarray:
    
    adi_combined = combine_fn(channels, axis=0)

    if out_path is not None:
        to_fits(adi_combined, out_path)

    return adi_combined