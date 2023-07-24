from astropy.io import ascii
import multiprocessing as mp
import numpy as np
from os.path import isfile
from typing import Callable
from vip_hci.fits import open_fits, write_fits

numworkers = 8      # number of worker threads/processes
numcomps = 3        # number of PCA components
everynthframe = 20  # number of frames 'n' selected from data cube
chunksize = 20
pxscale = 0.035

def init(data_paths: list[str], wavelengths_path: str, angles_path: str, channels: list[int]=..., frames: list[int]=...) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Get all data given the specified frames and channels to use.

    Returns a 4D data cube associated with all channels and frames, a list of
    angles that correspond to the frames, and a list of wavelengths that
    correspond to the channels.
    '''

    # >>>> figure out why angles are not the double precision as in the file

    # load data cube for each desired channel and frame as 4D cube
    cubes = loadall(data_paths)[:, frames].copy()

    # get list of wavelengths for proper calibration and scaling
    wavelengths = loadtbl(wavelengths_path, index=channels)

    # get list of angles for de-rotation
    angles = loadtbl(angles_path, index=frames)

    return cubes, wavelengths, angles

def loadtbl(path: str, index: list[int]=...) -> np.ndarray:
    table = ascii.read(path, format="no_header", data_start=0)
    data = table["col1"].data[index].copy()
    return data

# >>>> add function to just open data instead of doing it w/in redux fn
def loadone(path: str, verbose: bool=False) -> np.ndarray:
    return open_fits(path, verbose=verbose)

def loadall(paths: list[str], verbose: bool=False) -> np.ndarray:
    with mp.Pool(numworkers) as pool:
        return np.array(pool.starmap(loadone, zip(paths, np.repeat([verbose], len(paths)))))

# save numpy array to specified path as .FITS file
def to_fits(data: np.ndarray, path: str, overwrite: bool=True) -> None:
    if isfile(path) and overwrite or not isfile(path):
        write_fits(path, data)

# save numpy array to specified path as .npy file
def to_npy(data: np.ndarray, path: str, overwrite: bool=True) -> None:
    if isfile(path) and overwrite or not isfile(path):
        np.save(path, data)

def make_name(lib: str, algo: str, sub_type: str, first_chnl: str, last_chnl: str, ncomp: int=None, nskip_frames: int=None) -> str:
    if ncomp is None:
        ncomp = numcomps
    if nskip_frames is None:
        nskip_frames = everynthframe
    
    algo_text = algo if "PCA" not in algo else "PCA%03i"%ncomp
    name = f"{lib}{algo_text}-{sub_type}_{first_chnl}-{last_chnl}_skip{nskip_frames}"
    return name

def combine(channels: np.ndarray,
        combine_fn: Callable[[np.ndarray], np.ndarray]=np.median,
        out_path: str=None) -> np.ndarray:
    
    adi_combined = combine_fn(channels, axis=0)

    if out_path is not None:
        to_fits(adi_combined, out_path)

    return adi_combined