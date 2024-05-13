# vidojevica

 reduce_v10.py & reduce_v10_astrometry.py

Both reduce_v10.py and reduce_v10_astrometry.py reduce CCD images taken with Milankovic telescope mounted at the Astronomical Station Vidojevica. The difference is in obtaining the astrometrical solution: the first one uses python package twirl and the second one uses external sowtware Astrometry. 

Three folders are created in the parent folder: 
    1. calibrated # To store calibrated images
    2. astrometry # To store astrometrically solved calibrated images
    3. photometry # To store output tables with measured photometry

After proper calibration and obtained WCS solutions, photometry is measured using Photutils package defining apertures by either the fixed FWHM or the FWHM that can vary from image to image in the following way: aperture radius is 2.8*FWHM, annulus radius starting 5 pixels away from the aperture radius and spanning for 10 pixels. 

    ifFWHMvariable=False # for fixied value of FWHM and ifFWHMvariable=True to enable variation on FWHM
    ifBadPixels=False # Defects like cosmic rays and hot/dead pixels are removed if ifBadPixels=True or ifBadPixel=False the correction is omitted (default value).

--------------------------------------------------------------------------------------------------

reuce02.py

Not so basic data reduction. 

The code is written to use dithering larger than the object of interest to create a super-skyflat. 

Backgound of the single image is measured as a median excluding all detected objects and saved. The crucial step is the creation of a super-skyflat image as a median combination of all the frames with objects detected and ignored (replaced with nans). Before combining, science frames are normalized to their median values. This is done only for the purpose of creating a super-skyflat. As the final step, super-skyflat is multilied with the medain of each corresponding science frame and than subtracted from it (in this step science frames are not normalized, they are as they are after basic reduction: bias, dark, flat). 

This is the proper backgound subtraction if one can create a super-skyflat. Finally, science frames can be combined in Iraf, using imcombine. The corresponding procedure in python is not good enough - makes strange artifacts.  
