# vidojevica

Not so basic data reduction. 

The code is written to use dithering larger than the object of interest to create a super-skyflat. 

Backgound of the single image is measured as a median excluding all detected objects and saved. The crucial step is the creation of a super-skyflat image as a median combination of all the frames with objects detected and ignored (replaced with nans). Before combining, science frames are normalized to their median values. This is done only for the purpose of creating a super-skyflat. As the final step, super-skyflat is multilied with the medain of each corresponding science frame and than subtracted from it (in this step science frames are not normalized, they are as they are after basic reduction: bias, dark, flat). 

This is the proper backgound subtraction if one can create a super-skyflat. Finally, science frames can be combined in Iraf, using imcombine. The corresponding procedure in python is not good enough - makes strange artifacts.  
