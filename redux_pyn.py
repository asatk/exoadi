from matplotlib import pyplot as plt
import numpy as np
import os

from pynpoint import Pypeline, FitsReadingModule, PSFpreparationModule, \
    PcaPsfSubtractionModule, FalsePositiveModule, FakePlanetModule, \
    ContrastCurveModule, FitsWritingModule
    
from pynpoint.readwrite.attr_reading import ParangReadingModule

import redux_utils

def reduce_channel_pyn(datadir: str, datapaths: list[str], outdir: str=None, outpath: str=None):

    os.system("export OMP_NUM_THREADS=%i"%(redux_utils.numworkers))

    sep_tuple = (0., 0.8, 0.05)
    angle_tuple = (0.0, 360.0, 72.0)

    if outdir is None:
        outdir = "./out"

    nchnls = len(datapaths)

    # >>>> select only certain frames of data

    # >>>> find a way to set the attribute for PARANG without the new file
    tempanglesname = "tempangles.txt"
    tempangles = np.array([redux_utils.angles] * nchnls).flatten()
    with open(datadir + "/" + tempanglesname, "w") as anglesfile:
        anglesfile.write('\n'.join(str(i) for i in tempangles))

    pipeline = Pypeline(working_place_in=datadir,
                    input_place_in=datadir,
                    output_place_in=outdir)
    module_read = FitsReadingModule(name_in="read",
                    filenames=datapaths,
                    image_tag='read')
    module_parang = ParangReadingModule(name_in="angs",
                    data_tag='read',
                    file_name=tempanglesname)
    module_prep = PSFpreparationModule(name_in='prep',
                    image_in_tag='read',
                    image_out_tag='prep',
                    mask_out_tag=None,
                    norm=False,
                    resize=None,
                    cent_size=0.2,
                    edge_size=None)
    module_psf = PSFpreparationModule(name_in='prep2',
                    image_in_tag='read',
                    image_out_tag='psf',
                    mask_out_tag=None,
                    norm=False,
                    resize=None,
                    cent_size=None,
                    edge_size=0.5)
    module_pcasub = PcaPsfSubtractionModule(
                    pca_numbers=range(1, redux_utils.numcomps + 1),
                    name_in='pca',
                    images_in_tag='prep',
                    reference_in_tag='prep',
                    res_mean_tag='pca_mean',
                    res_median_tag='pca_median',
                    basis_out_tag='pca_basis',
                    subtract_mean=True)
    module_fp = FalsePositiveModule(
                    name_in='fp',
                    image_in_tag='pca_median',
                    snr_out_tag='snr',
                    position=(12, 41),
                    aperture=5*0.027,
                    ignore=True,
                    optimize=False)
    module_fakeplanet = FakePlanetModule(
                    name_in='fake',
                    image_in_tag='prep',
                    psf_in_tag='psf',
                    image_out_tag='removed',
                    # position=,
                    # magnitude=,
                    psf_scaling=-1.0,
                    interpolation='spline')
    module_ccurve = ContrastCurveModule(
                    name_in='ccurve',
                    image_in_tag='removed',
                    psf_in_tag='psf',
                    contrast_out_tag='limits',
                    separation=sep_tuple,
                    angle=angle_tuple,
                    threshold=('sigma', 5.0),
                    # psf_scaling=,
                    # aperture=,
                    # pca_number=,
                    # cent_size=,
                    # edge_size=,
                    # extra_rot=,
                    residuals='cc_median',
                    snr_inject=100.)
    module_write1 = FitsWritingModule(
                    name_in='write1',
                    data_tag='pca_median',
                    file_name='pca_median.fits',
                    data_range=None,
                    overwrite=True,
                    subset_size=None)
    module_write2 = FitsWritingModule(
                    name_in='write2',
                    data_tag='pca_mean',
                    file_name='pca_mean.fits',
                    data_range=None,
                    overwrite=True,
                    subset_size=None)

    pipeline.add_module(module_read)
    pipeline.add_module(module_parang)
    pipeline.add_module(module_prep)
    pipeline.add_module(module_pcasub)
    pipeline.add_module(module_psf)
    pipeline.add_module(module_fp)
    pipeline.add_module(module_fakeplanet)
    pipeline.add_module(module_ccurve)
    pipeline.add_module(module_write1)
    pipeline.add_module(module_write2)
    pipeline.run()

    res_median = pipeline.get_data('pca_median')
    redux_utils.savedata(res_median[0,], outpath)
    
    pixscale = pipeline.get_attribute('pca_median', 'PIXSCALE')
    size = pixscale * res_median.shape[-1]/2

    plt.imshow(res_median[redux_utils.numcomps], origin='lower', extent=[size, -size, -size, size])
    plt.xlabel('RA offset (arcsec)', fontsize=14)
    plt.ylabel('Dec offset (arcsec)', fontsize=14)
    cb = plt.colorbar()
    cb.set_label('Flux (ADU)', size=14.)
    plt.show()

if __name__ == "__main__":

    # --- DATA INFO --- #
    firstchannelnum = 40
    lastchannelnum = 80
    channelnums = list(range(firstchannelnum, lastchannelnum))
    nchnls = len(channelnums)
    # --- DATA INFO --- #

    # --- PATHS --- #
    datadir = "./data"
    datapath = datadir + "/" + "005_center_multishift/wl_channel_%05i.fits"
    datapaths = [datapath%(channelnum) for channelnum in channelnums]

    outdir = "./out"
    outcombinedpath = "./out/pyn_PCA%03i_%05i_%05i.fits"%(redux_utils.numcomps, firstchannelnum, lastchannelnum)
    # --- PATHS --- #

    reduce_channel_pyn(datadir, datapaths, outdir=outdir, outpath=outcombinedpath)