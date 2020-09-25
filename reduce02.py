import sys
import os
import glob
#import fnmatch

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils import make_source_mask
#from astropy.visualization import simple_norm
#from astropy import stats as astrostats
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
#from astropy.visualization import SqrtStretch
#from astropy.visualization.mpl_normalize import ImageNormalize
#from astropy.stats import SigmaClip
#from photutils import Background2D, MedianBackground
#from photutils import BkgZoomInterpolator
from astropy.visualization import simple_norm

path = "/Users/anavudragovic/vidojevica/15_09_2020/" # path the directory containing all images
bias_names="bias*.fit" # how to identify the bias images by file name. 
dark_for_flats_names="dark*5sec*fit" #dark images for flats.
dark_for_science_names="dark600s*fit" # dark images for science images.
flats_names="flat*.fit"
science_names="NGC4589-000*_L.fit" # how to identify the science images by file name.
# I did this naming convention because the telescope pipeline doesn't include astrometry, which therefore needed
# to be done separately. To keep everything in one folder, I added "wcs" to the file name and search for it
# in this name query. Of course there are smarter ways to do this, but it works.


os.chdir(path)


biases=[]
for file in glob.glob(bias_names):
    biases.append(fits.open(file)[0].data)
biases=np.array(biases)
bias=np.median(biases,axis=0)
#hdu = fits.PrimaryHDU(bias)
#hdu.writeto(path+'bias.fits',overwrite=True)
norm = simple_norm(bias,'sqrt',percent=99)
plt.title("master bias")
plt.imshow(bias,norm=norm,cmap = 'Greys', origin='lower')
plt.show()


# Dark current in 5 sec
darks_flat=[]
for file in glob.glob(dark_for_flats_names):
    darks_flat.append(fits.open(file)[0].data-bias)
darks_flat=np.array(darks_flat)
dark_current_5s=np.median(darks_flat,axis=0)
norm = simple_norm(dark_current_5s,'sqrt',percent=99)
plt.title("Dark current in 5 sec frame")
plt.imshow(dark_current_5s, norm=norm, cmap = 'Greys', origin='lower')
plt.show()

# Dark current in 600 sec
darks_science=[]
for file in glob.glob(dark_for_science_names):
    darks_science.append(fits.open(file)[0].data-bias)
darks_science=np.array(darks_science)
dark_current_600s=np.median(darks_science,axis=0)
norm = simple_norm(dark_current_600s,'sqrt',percent=99)
plt.title("Dark current in 600 sec frame")
plt.imshow(dark_current_600s, norm=norm, cmap = 'Greys', origin='lower')
plt.show()

flats_L=[]

for file in glob.glob(flats_names):
        flat_db=fits.open(file)[0].data-bias-dark_current_5s
        flat_norm=flat_db/np.median(flat_db)
        flats_L.append(flat_norm)

flats_L=np.array(flats_L)
masterflat_L=np.median(flats_L,axis=0)
print("L: ",len(flats_L),np.shape(flats_L))
norm = simple_norm(masterflat_L,'sqrt',percent=99)
plt.title("master flat L")
plt.imshow(masterflat_L, norm=norm, cmap = 'Greys', origin='lower')
plt.show()


# Make supersky flat
hdu_list=[]
sky_flats=[]
num_images = len(glob.glob(science_names))
science_median=[]
for file in glob.glob(science_names):
    hdul=fits.open(file)
    hdu = hdul[0]
    exptime=hdu.header['EXPTIME']
    # calibrate the data
    science_db=hdu.data-bias-dark_current_600s/600*exptime
    science=science_db/masterflat_L
    mask = make_source_mask(science, nsigma=2, npixels=5, dilate_size=11)
    mean, median, std = sigma_clipped_stats(science, sigma=3.0, mask=mask)
    science_median.append(median)
    science=science/median
    data_masked = np.ma.masked_where(mask, science, True)
    science[data_masked.mask == True] = np.nan
    sky_flats.append(science)
    hdul.close()

sky_flat=np.nanmedian(sky_flats,axis=0)
norm = simple_norm(sky_flat,'sqrt',percent=99)
plt.title("Super sky flat L")
plt.imshow(sky_flat, norm=norm, cmap = 'Greys', origin='lower')
plt.show()

# Calibrate images and correct them for sky flat
hdu_list=[]
i=0
for file in glob.glob(science_names):
    hdu=fits.open(file)[0]
    exptime=hdu.header['EXPTIME']
    # calibrate the data
    science_db=hdu.data-bias-dark_current_600s/600*exptime
    science=science_db/masterflat_L
    science=science-science_median[i]*sky_flat

    hdu.data=science
    hdu_list.append(hdu)
    wcs = WCS(hdu.header)
    hdu.header.update(wcs.to_header())
    hdu.writeto(path+"cal" + file, overwrite=True)
    i=i+1


# ----------------------------------------------------------------

