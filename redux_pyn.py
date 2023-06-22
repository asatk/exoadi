from matplotlib import pyplot as plt
import numpy as np
import os

from pynpoint import Pypeline, FitsReadingModule, PSFpreparationModule, PcaPsfSubtractionModule
from pynpoint.readwrite.attr_reading import ParangReadingModule

import redux_utils

def reduce_channel_pyn(datadir: str, datapaths: list[str], outdir: str=None, outpath: str=None):

    os.system("export OMP_NUM_THREADS=%i"%(redux_utils.numworkers))

    if outdir is None:
        outdir = "./out"

    nchnls = len(datapaths)

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

    # module = PSFpreparationModule(name_in='prep',
    #                               image_in_tag='stack',
    #                               image_out_tag='prep',
    #                               mask_out_tag=None,
    #                               norm=False,
    #                               resize=None,
    #                               cent_size=0.15,
    #                               edge_size=1.1)

    # pipeline.add_module(module)

    module_pcasub = PcaPsfSubtractionModule(pca_numbers=[redux_utils.numcomps, ],
                                    name_in='pca',
                                    images_in_tag='read',
                                    reference_in_tag='read',
                                    res_median_tag='residuals')

    pipeline.add_module(module_read)
    pipeline.add_module(module_parang)
    pipeline.add_module(module_pcasub)

    pipeline.run()

    residuals = pipeline.get_data('residuals')
    # pixscale = pipeline.get_attribute('residuals', 'PIXSCALE')
    # size = pixscale * residuals.shape[-1]/2

    redux_utils.savedata(residuals[0,], outpath)

    # plt.imshow(residuals[0, ], origin='lower', extent=[size, -size, -size, size])
    # plt.xlabel('RA offset (arcsec)', fontsize=14)
    # plt.ylabel('Dec offset (arcsec)', fontsize=14)
    # cb = plt.colorbar()
    # cb.set_label('Flux (ADU)', size=14.)
    # plt.show()

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