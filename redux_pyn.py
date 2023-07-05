import faulthandler
from matplotlib import pyplot as plt
import numpy as np
import os

from pynpoint import Pypeline, FitsReadingModule, PSFpreparationModule, \
    PcaPsfSubtractionModule, FalsePositiveModule, SimplexMinimizationModule, \
    FakePlanetModule, ContrastCurveModule, FitsWritingModule
    
from pynpoint.readwrite.attr_reading import ParangReadingModule

import redux_utils

def reduce_channel_pyn(data_dir: str, data_paths: list[str], out_dir: str=None, out_path: str=None, working_dir: str=None):

    os.system("export OMP_NUM_THREADS=%i"%(redux_utils.numworkers))

    sep_tuple = (0., 0.8, 0.05)
    angle_tuple = (0.0, 360.0, 72.0)

    if out_dir is None:
        out_dir = "out"

    if working_dir is None:
        working_dir = "pyn"

    nchnls = len(data_paths)
    # pca_nums = list(range(1, redux_utils.numcomps + 1))
    # pca_num = redux_utils.numcomps
    pca_nums = [5]
    pca_num = 5

    mask_inner = 0.2
    mask_outer = 0.35

    posn_planet = (12, 41)
    dmag_planet = 6.35

    # figure out which apt corresponds to what
    pixscale = 0.027
    apt_pl = 5 * pixscale
    apt_psf = 10 * pixscale
    apt_cc = 0.2


    tolerance = 0.1
    offset = 0.1

    # >>>> select only certain frames of data cube?

    # >>>> find a way to set the attribute for PARANG without the new file IFS?
    tempanglesname = "tempangles.txt"
    tempangles = np.array([redux_utils.angles] * nchnls).flatten()
    with open(data_dir + "/" + tempanglesname, "w") as anglesfile:
        anglesfile.write('\n'.join(str(i) for i in tempangles))

    pipeline = Pypeline(working_place_in=working_dir,
                    input_place_in=data_dir,
                    output_place_in=out_dir)
    module_read = FitsReadingModule(name_in="read",
                    filenames=data_paths,
                    image_tag='imgs')
    module_parang = ParangReadingModule(name_in="angs",
                    data_tag='imgs',
                    file_name=tempanglesname,
                    overwrite=True)
    module_prep = PSFpreparationModule(name_in='prep',
                    image_in_tag='imgs',
                    image_out_tag='masked',
                    mask_out_tag=None,
                    norm=False,
                    resize=None,
                    cent_size=mask_inner,
                    edge_size=None)
    module_psf = PSFpreparationModule(name_in='prep2',
                    image_in_tag='imgs',
                    image_out_tag='psf',
                    mask_out_tag=None,
                    norm=False,
                    resize=None,
                    cent_size=None,
                    edge_size=mask_outer)
    module_pcasub = PcaPsfSubtractionModule(name_in='pca',
                    pca_numbers=pca_nums,
                    images_in_tag='masked',
                    reference_in_tag='masked',
                    res_mean_tag='pca_mean',
                    res_median_tag='pca_median',
                    basis_out_tag='pca_basis',
                    subtract_mean=True,
                    processing_type='ADI')
    # module_fp = FalsePositiveModule(name_in='fp',
    #                 image_in_tag='pca_median',
    #                 snr_out_tag='snr',
    #                 position=posn_planet,
    #                 aperture=5*0.027,
    #                 ignore=True,
    #                 optimize=True,
    #                 offset=offset)
    # module_simplex = SimplexMinimizationModule(name_in="simplex",
    #                 image_in_tag="imgs",
    #                 psf_in_tag="psf",
    #                 res_out_tag="simplex_res",
    #                 flux_position_tag="fluxpos",
    #                 position=posn_planet,
    #                 magnitude=dmag_planet,
    #                 psf_scaling=-1.,
    #                 merit="gaussian",
    #                 aperture=10*0.027,
    #                 sigma=0.,
    #                 tolerance=tolerance,
    #                 pca_number=pca_nums,
    #                 cent_size=mask_inner,
    #                 edge_size=None,
    #                 residuals="median",
    #                 reference_in_tag=None,
    #                 offset=offset)
    # module_fakeplanet = FakePlanetModule(name_in="fake",
    #                 image_in_tag="masked",
    #                 psf_in_tag="psf",
    #                 image_out_tag="removed",
    #                 position=posn_planet,
    #                 magnitude=dmag_planet,
    #                 psf_scaling=-1.0,
    #                 interpolation="spline")
    # module_ccurve = ContrastCurveModule(name_in="ccurve",
    #                 image_in_tag="removed",
    #                 psf_in_tag="psf",
    #                 contrast_out_tag="limits",
    #                 separation=sep_tuple,
    #                 angle=angle_tuple,
    #                 # threshold=("sigma", 5.0),
    #                 threshold=('fpf', 2.87e-7),
    #                 psf_scaling=1.,
    #                 aperture=0.2,
    #                 pca_number=pca_num,
    #                 cent_size=mask_inner,
    #                 edge_size=None,
    #                 residuals="median",
    #                 snr_inject=100.)
    module_write1 = FitsWritingModule(name_in='write1',
                    data_tag='pca_median',
                    file_name='pyn_pca_median.fits',
                    data_range=None,
                    overwrite=True,
                    subset_size=None)
    module_write2 = FitsWritingModule(name_in='write2',
                    data_tag='pca_mean',
                    file_name='pyn_pca_mean.fits',
                    data_range=None,
                    overwrite=True,
                    subset_size=None)

    pipeline.add_module(module_read)
    pipeline.add_module(module_parang)
    pipeline.add_module(module_prep)
    pipeline.add_module(module_pcasub)
    pipeline.add_module(module_psf)
    # pipeline.add_module(module_fp)
    # pipeline.add_module(module_simplex)
    # pipeline.add_module(module_fakeplanet)
    # pipeline.add_module(module_ccurve)
    pipeline.add_module(module_write1)
    pipeline.add_module(module_write2)
    pipeline.run()

    res_median = pipeline.get_data('pca_median')
    redux_utils.savedata(res_median[0,], out_path%("median"))

    res_mean = pipeline.get_data('pca_mean')
    redux_utils.savedata(res_mean[0,], out_path%("mean"))
    
    pixscale = pipeline.get_attribute('pca_median', 'PIXSCALE')
    size = pixscale * res_median.shape[-1]/2

    plt.imshow(res_median[redux_utils.numcomps], origin='lower', extent=[size, -size, -size, size])
    plt.xlabel('RA offset (arcsec)', fontsize=14)
    plt.ylabel('Dec offset (arcsec)', fontsize=14)
    cb = plt.colorbar()
    cb.set_label('Flux (ADU)', size=14.)
    plt.show()

if __name__ == "__main__":

    faulthandler.enable()

    # --- DATA INFO --- #
    firstchannelnum = 45
    lastchannelnum = 74
    channelnums = list(range(firstchannelnum, lastchannelnum))
    nchnls = len(channelnums)
    # --- DATA INFO --- #

    # --- PATHS --- #
    data_dir = "data"
    data_path = data_dir + "/" + "005_center_multishift/wl_channel_%05i.fits"
    data_paths = [data_path%(channelnum) for channelnum in channelnums]

    out_dir = "out"
    outcombined_path = "out/pyn_PCA%03i_%05i_%05i_%s.fits"%(redux_utils.numcomps, firstchannelnum, lastchannelnum, "%s")

    working_dir = "pyn"
    # --- PATHS --- #

    reduce_channel_pyn(data_dir, data_paths, out_dir=out_dir,
                       out_path=outcombined_path, working_dir=working_dir)