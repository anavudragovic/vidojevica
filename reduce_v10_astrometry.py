# conda create -c conda-forge -n spyder-env spyder numpy scipy pandas matplotlib sympy cython astropy photutils ccdproc
# conda activate spyder-env
# import numpy, astropy, photutils, ccdproc
# print(numpy.__version__) # 1.26.0
# print(astropy.__version__) # 5.3.4
# print(photutils.__version__) # 1.10.0
# print(ccdproc.__version__) # 2.4.1
########################## FUNCTIONS ###########################################

##################################### KALIBRACIJA ##############################


def makeMasterBIAS(file_list):
  
  frame_list = []
  for fl in file_list:
    with fits.open(fl) as hdul:
      frame_list.append(hdul[0].data)
  frame_arr = np.array(frame_list)
  
  mFrame = np.median(frame_arr, axis=0)
  
  return mFrame
  
  
  
def makeMasterDARK(file_list, mBias):
  
  frame_list = []
  for fl in file_list:
    with fits.open(fl) as hdul:
      frame_list.append(hdul[0].data - mBias)
  frame_arr = np.array(frame_list)
  
  mFrame = np.median(frame_arr, axis=0)
  
  return mFrame
  
  

def makeMasterFLAT(file_list, mBias, mDark_dic):
  
  frame_list = []
  for fl in file_list:
    with fits.open(fl) as hdul:
      expt_flat = float(hdul[0].header['EXPTIME'])
      expt_dark_max = np.max(list(mDark_dic.keys()))
      
      # BIAS + DARK correction
      if expt_flat in mDark_dic: # Check the line below for mBias:
        frame_cal = hdul[0].data - mBias - mDark_dic[expt_flat]
      elif expt_dark_max > expt_flat:
        scale = expt_flat / expt_dark_max
        frame_cal = hdul[0].data - mBias - scale * mDark_dic[expt_dark_max]
      else:
        frame_cal = hdul[0].data - mBias
      
      # flats normalization
      frame_cal = frame_cal / np.median(frame_cal)
      
      # append
      frame_list.append(frame_cal)  
  
  # turn into array
  frame_arr = np.array(frame_list)
  
  # average over 0 axis
  mFrame = np.median(frame_arr, axis=0)
  
  return mFrame







def process_makeMasterSKYFLAT(fl, mBias, mDark_dic, mFlat_dic):
  
  
    with fits.open(fl, mode='update') as hdul:
        #print("Making sky flat:")
        header = hdul[0].header
        data = hdul[0].data
        filt_skyflat = str(header['FILTER'])
        expt_skyflat = float(header['EXPTIME'])
        expt_dark_max = np.max(list(mDark_dic.keys()))
        
        # BIAS + DARK korekcija
        if expt_skyflat in mDark_dic:
            frame_cal = data - mBias - mDark_dic[expt_skyflat]
        elif expt_dark_max > expt_skyflat:
            scale = expt_skyflat / expt_dark_max
            frame_cal = data - mBias - scale * mDark_dic[expt_dark_max]
        else:
            frame_cal = data - mBias
   
        # FLAT korekcija
        if filt_skyflat in mFlat_dic:
            frame_cal = frame_cal / mFlat_dic[filt_skyflat]
            frame_cal = frame_cal / expt_skyflat
       
        # mask all point sources with the surroundings usign convolution with
        # Gaussain filter that expands the masks created
        #mask = make_source_mask(frame_cal, nsigma=2, npixels=5, dilate_size=11)
        # NEW
        mean, _, std = sigma_clipped_stats(frame_cal)

        # subtract the background
        frame_cal = frame_cal - mean
        
        # detect the sources
        threshold = 3. * std
        kernel = make_2dgaussian_kernel(3.0, size=3)  # FWHM = 3.
        convolved_data = convolve(frame_cal, kernel)
        segm = detect_sources(convolved_data, threshold, npixels=5)
        footprint = circular_footprint(radius=10)
        mask = segm.make_source_mask(footprint=footprint)
        # background statistics
        mean, median, std = sigma_clipped_stats(frame_cal, sigma=3.0, mask=mask)
        # update header with background statistics - needed for sky flat correction
        header['MEAN'] = (mean, 'Mean value after 3 sigma clip with masked objects')
        header['MEDIAN'] = (median, 'Median value after 3 sigma clip with masked objects')
        header['STDEV'] = (std, 'Standard deviation after 3 sigma clip with masked objects')
        hdul.flush()
     
        # normalize to unity
        frame_cal = frame_cal / median
        # mask all objects brighter than 3-sigma
        frame_cal_masked = np.ma.masked_array(frame_cal, mask=mask).filled(np.nan).astype(np.float32)
       
        # if less than 10 percent is masked than return sky_flat image
        # otherwise it is not suitable for applying correction and return none
        if (np.sum(np.isnan(frame_cal_masked)) / frame_cal_masked.size * 100. < 10.):
            return frame_cal_masked
        else:
            return None


def makeMasterSKYFLAT_multiprocess(skyflatNms_list, mBias, mDark_dic, mFlat_dic):
  
    if len(skyflatNms_list) > 50:
        n_processes = mp.cpu_count() - 1 # ostavi jedan za druge rabote
        pool = mp.Pool(processes=n_processes)

        # pokreni paralelni rad
        results = []
        for fl in skyflatNms_list:
            results.append(pool.apply_async(process_makeMasterSKYFLAT, args=(fl, mBias, mDark_dic, mFlat_dic)))
        pool.close()
        pool.join()

        # iscitaj rezultat
        OK_skyflat_list = []
        for res in results:
            frame_cal_masked = res.get()
            if frame_cal_masked is not None:
                OK_skyflat_list.append(frame_cal_masked)
                
        # izracunaj median
        if len(OK_skyflat_list) > 50:   # promeni u neki broj koji odgovara RAMu 
          
          filename = 'memmap_file' 
          n_files = len(OK_skyflat_list)   
          memmap_shape = (n_files, OK_skyflat_list[0].shape[0], OK_skyflat_list[0].shape[1])
          memmap_array = np.memmap(filename, dtype='float32', mode="w+", shape=memmap_shape)
          
          # kopiraj podatke iz 2D nizova u memmap datoteku
          for i in range(n_files):
            memmap_array[i, :, :] = OK_skyflat_list[i]

          # izračunaj np.nanmedian
          median_array = np.nanmedian(memmap_array, axis=0)
          
          os.system("rm -f "+filename)
          
          # vrati
          return median_array
        
        else:
          
          # izračunaj np.nanmedian
          median_array = np.nanmedian(OK_skyflat_list, axis=0)
          
          # vrati
          return median_array
          
    return None


  
def process_calibrate(args):
  
  fl, mBias, mDark_dic, mFlat_dic, mSkyflat_dic = args
  
  print(f'-----> Calibrate {fl} file')
  
  with fits.open(fl) as hdul:
    header = hdul[0].header
    data = hdul[0].data
    
  filt_light = str(header['FILTER'])
  expt_light = float(header['EXPTIME'])
  expt_dark_max = np.max(list(mDark_dic.keys()))

  # BIAS + DARK korekcija
  if expt_light in mDark_dic:
      frame_cal = data - mBias - mDark_dic[expt_light]
  elif expt_dark_max > expt_light:
      scale = expt_light / expt_dark_max
      frame_cal = data - mBias - scale * mDark_dic[expt_dark_max]
  else:
      frame_cal = data - mBias
  
  # FLAT korekcija
  if filt_light in mFlat_dic:
      frame_cal = frame_cal / mFlat_dic[filt_light]
  
  # SKYFLAT korekcija
  if mSkyflat_dic[filt_light] is not None:
    median_light = float(header['MEDIAN'])
    frame_cal = frame_cal / expt_light
    frame_cal = frame_cal - median_light * mSkyflat_dic[filt_light]
  print(calibFolderNm)
  
  if ifPlotCalibratedFrames:    
      # Plot calibrated frame
      interval = ZScaleInterval()
      zmin, zmax = interval.get_limits(frame_cal)
      fig, ax = pl.subplots(1,1, figsize=(12,12))
      ax.set_title(os.path.basename(fl))
      ax.imshow(frame_cal, origin='lower', cmap='Greys_r', vmin=zmin, vmax=zmax, interpolation='nearest')

  outFlNm = fl.split(".")[0]+"_cal.fit"
  header['BSCALE'] = 1.
  header['BZERO'] = 0.
  primHDU = fits.PrimaryHDU()
  primHDU.header = header
  primHDU.data = np.float32(frame_cal)
  hdulist = fits.HDUList([primHDU])
  hdulist.writeto(os.path.join(calibFolderNm, outFlNm), overwrite=True)
  
      

  
def calibrate_multiprocess(lightNms_list, mBias, mDark_dic, mFlat_dic, mSkyflat_dic):
      
  # use more threads   
  with concurrent.futures.ThreadPoolExecutor() as executor:
    args = ((fl, mBias, mDark_dic, mFlat_dic, mSkyflat_dic) for fl in lightNms_list)
    executor.map(process_calibrate, args)
  
 
 
 

##################################### ASTROMETRY ############################## 
 
def getCPUtemp():
  
  temperatures = psutil.sensors_temperatures()
  temps_list = [r.current for r in temperatures['coretemp']]

  return temps_list



def doAstrometry(FITSfls_list):
  
  # sakupi FITSnm, RA, DEC
  lines = []
  for fl in FITSfls_list:
    with fits.open(fl) as hdul:
      header = hdul[0].header
    lines.append(f"{fl} {header['OBJCTRA'].replace(' ',':')} {header['OBJCTDEC'].replace(' ',':')}")  
  
  # number of FITS files
  Nfits = len(lines)
  # define number of threades
  Ncores = psutil.cpu_count() - 1
  # number of loops
  Nloops = int(math.ceil(Nfits/float(Ncores))) # ceil zaokruzuje broj navise
  k = 0
  for i in range(Nloops):
    
    with open("tmp", "w") as outFl:
      for j in range(int(Ncores)):
        if k >= Nfits:
          continue
        else:  
          FITSnm, RA, DEC = lines[k].split()
          print("astrometry:",FITSnm, RA, DEC)
          FITSout=FITSnm.replace("cal.fit","cal_wcs.fit" ).replace("calibrated","plate_solved")
          outFl.write("solve-field "+FITSnm+" --ra "+RA+" --dec "+DEC+" --radius 0.2 --scale-units arcsecperpix --scale-low 0.38 --scale-high 0.40 --crpix-center -p -N "+FITSout+" -O -I noneI.fits -M none.fits -R none.fits -B none.fits -P none -k none -U none -y -S none --axy noneaxy --wcs bla.wcs\n")
          k += 1

    cmd = "parallel -j "+str(Ncores)+" < tmp"
    os.system(cmd)
    
    CPUtemps_list = getCPUtemp()
    
    if ifPauseProcess: 
      if np.any(np.array(CPUtemps_list) > 80.):
        print ("If any of the core temperatures > 80C the sleep time is set to 30 sec")
        time.sleep(30)
      else:
        print ("If any of the core temperatures < 80C the sleep time is set to 5 sec")
        time.sleep(5)




##################################### PHOTOMETRY ############################## 


    
  
def doPhotometry(args):
  
    fl, prefix = args
    #print("Aperture photometry on ", prefix, ': ', fl)
    with fits.open(fl) as hdul:
      data = hdul[0].data
      header = hdul[0].header
      #data = data / header['EXPTIME'] # images normalized to 1 sec exptime: ADU/sec
      wcs = WCS(header)
      object_name = header['OBJECT']
      
      #print(object_name)
      mean, median, std = sigma_clipped_stats(data, sigma=3.0)
      data -= median
      size = 27 # size of a star image and size of a border to exclude from edges
      targets_list=object_name + '_stars.txt'  
      
      if os.path.exists(targets_list) and os.path.getsize(targets_list) > 0:
          print('photometry task:', fl," -> Target file exists.")
          sources=pd.read_csv(targets_list,names=['RAJ2000','DECJ2000','id'],sep=" ")
          flag = np.zeros(len(sources),dtype='int')
          pixel_coo = wcs.wcs_world2pix(sources['RAJ2000'], sources['DECJ2000'], 0)
          xpos, ypos = pixel_coo[0], pixel_coo[1]
          flux_peak = data[np.rint(ypos).astype(int),np.rint(xpos).astype(int)]
          #world = wcs.wcs_pix2world(np.transpose(pixel_coo), 0)
      else:
          #print('photometry task:', fl," -> Target file doesn't exist.")
          mask_edges = np.zeros(data.shape, dtype=bool)
          mask_edges[size:header['NAXIS1']-size, size:header['NAXIS2']-size] = True # exclude image borders
          # daofind = IRAFStarFinder(fwhm=2., threshold=3.*std, sharplo=0.5, sharphi=1., roundlo=-1, roundhi=1, exclude_border=True, minsep_fwhm=3)
          # sources = daofind(data-median, mask=~mask_edges)
          # positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
          # pixel_coo = sources['xcentroid'], sources['ycentroid']
          # for col in sources.colnames:  
          #      if col not in ('id', 'npix'):
          #          sources[col].info.format = '%.1f'  # for consistent table output
          threshold = 5.0 * std
          from photutils.detection import find_peaks
          sources = find_peaks(data, threshold, box_size=11, mask=~mask_edges)
          flag = np.zeros(len(sources),dtype='int')
          positions = np.transpose((sources['x_peak'], sources['y_peak']))
          pixel_coo = sources['x_peak'], sources['y_peak']
          flux_peak = sources['peak_value']  #* header['EXPTIME'] # if images are divided by exptime uncomment
          # for col in sources.colnames:  
          #      if col not in ('id', 'npix'):
          #          sources[col].info.format = '%.1f'  # for consistent table output

      #sources.pprint(max_width=76) 
      fwhm=4.

      x_init, y_init = pixel_coo
      x, y = centroid_sources(data, x_init, y_init, box_size=11,
                              centroid_func=centroid_2dg)
      if ifFWHMvariable:
           fwhm_i = []
           # (1) measure FWHM by combining comparison stars excluding target
           # or simply combine all except for the first star found in the image
           # (2) Model each object with a 2D Gaussian and measure FWHM 
           gauss = fitting.LevMarLSQFitter()
           for i in range(0,len(x[1:])):
               cutout = Cutout2D(data-median, [x[i+1],y[i+1]], size)
               # Try to model each object
               star=cutout.data/np.mean(cutout.data)
               y0, x0 = np.unravel_index(np.argmax(star), star.shape)
               sigma = np.std(star)
               amp = np.max(star)
               my_gauss = models.Gaussian2D(amp, x0, y0, sigma, sigma)
               yi, xi = np.indices(star.shape)
               g = gauss(my_gauss, xi, yi, star)
               fwhm_i.append((g.x_fwhm + g.y_fwhm)/2.)
           # Optimal FWHM is a mean/median value    
           fwhm = np.mean(fwhm_i)
           #print('fwhm=',fwhm)
      else:
           fwhm = 4.

      pixel_coo = x, y
      positions = np.transpose(pixel_coo)
      apertures = CircularAperture(positions, r=2.8*fwhm)
      annulus_aperture = CircularAnnulus(positions, r_in=2.8*fwhm+5, r_out=2.8*fwhm+15)
      sigclip = SigmaClip(sigma=3.0, maxiters=10)
      bkg_stats = ApertureStats(data, annulus_aperture, sigma_clip=sigclip)
      #total_flux = ApertureStats(data, apertures)
      aper_stats_bkgsub = ApertureStats(data, apertures, local_bkg=bkg_stats.median)
      flux=aper_stats_bkgsub.sum
      #flux_high=total_flux.sum-(bkg_stats.median-bkg_stats.std/2)*apertures.area
      #flux_low=total_flux.sum-(bkg_stats.median+bkg_stats.std/2)*apertures.area
      flux_err = np.sqrt(flux[flux > 0] + apertures[flux > 0].area * bkg_stats[flux > 0].std**2 * (1+apertures[flux > 0].area/annulus_aperture[flux > 0].area))
      mag = -2.5*np.log10(flux[flux > 0])
      mag_err = 1.0857 * flux_err / flux[flux > 0]
      positions = positions[flux>0]
      xpos = pixel_coo[0][flux>0]
      ypos = pixel_coo[1][flux>0]
      world = wcs.wcs_pix2world(positions, 0)
      flag[flux_peak>60000]=1  
      flag = flag[flux>0]
      flux_peak = flux_peak[flux>0]
      flux = flux[flux > 0]
      # Write an output ascii file with aperture photometry             
      # The case when list of comparison stars in provided
      if os.path.exists(targets_list) and os.path.getsize(targets_list) > 0:
            photfile = prefix+'_photometry.txt'
            with open(targets_list, "r") as f:
               num_stars = len(f.readlines())
               f.close()
            if not (os.path.exists(photfile) and os.path.getsize(photfile) > 0):
                 headline = ['FILENAME','MJD-HELIO','FILTER','DATE-OBS']
                 for i in range(1,num_stars):
                     headline.append('TmC'+str(i))
                 for i in range(1,num_stars):
                     headline.append('TmC'+str(i)+'_err')
                 for i in range(1,num_stars-1):
                     for j in range(i+1,num_stars):
                         headline.append('C'+str(i)+'mC'+str(j))
                 for i in range(1,num_stars-1):
                     for j in range(i+1,num_stars):
                          headline.append('C'+str(i)+'mC'+str(j)+'_err')
                 headline.append('\n')
            else:
                 headline=[]
                 
            with open(prefix+'_photometry.txt', mode='a+') as outname:
                magdiff=np.zeros(int((num_stars-1)*(num_stars-2)/2),dtype=float)
                magdiff_err=np.zeros(int((num_stars-1)*(num_stars-2)/2),dtype=float)
                k=0 
                for i in range(1,num_stars-1):
                    for j in range(i+1,num_stars):
                         magdiff[k] = mag[i]-mag[j]
                         magdiff_err[k] = np.sqrt(mag_err[i]**2 + mag_err[j]**2)
                         k+=1
                #print(prefix+'_photometry.txt target file exists.')
                magnitudes=np.concatenate((mag[0]-mag[1:],np.sqrt(mag_err[0]**2 + mag_err[1:]**2),magdiff,magdiff_err))
                txtline=[]                 
                txtline=[fl,str(header['JD-HELIO']-2400000.5),str(header['FILTER']),str(header['DATE-OBS'])]
                magline=magnitudes.tolist()
                outline=headline+txtline+magline
                outline.append('\n')
                outname.writelines("%s " % item for item in outline)
                outname.close()
            
            table = Table()
            #table['RAJ2000'] = world[:,0][d2d.arcsec<1]
            table['RAJ2000'] = world[:,0]
            table['DECJ2000'] = world[:,1]
            table['flux_peak'] = flux_peak
            table['flux'] = flux
            table['flux_err'] = flux_err
            table['mag'] = mag
            table['mag_err'] = mag_err
            table['AIRMASS'] = header['AIRMASS']
            for col in table.colnames:
                table[col].info.format = '%.4f'
            table['FILTER'] = header['FILTER']  
            table['MJD-HELIO'] = header['JD-HELIO']-2400000.5
            table['DATE-OBS'] = header['DATE-OBS']
            table['flag'] = flag
            # Output table name will include FWHM used for aperture photometry
            table.write(fl.replace("fit","txt").replace("astrometry","photometry"), format='ascii', overwrite=True)
      else:
            #print('The case when list of comparison stars in NOT provided')
            #print('Save instrumental magnitudes and magnitudes calibrated to GAIA photometry')
            phottable = Table()
            #phottable['xpix'] = xpos
            #phottable['ypix'] = ypos
            phottable['RAJ2000'] = world[:,0]
            phottable['DECJ2000'] = world[:,1]
            phottable['flux_max'] = flux_peak
            phottable['flux'] = flux
            phottable['flux_err'] = flux_err
            phottable['mag'] = mag
            phottable['mag_err'] = mag_err
            #phottable['flux_high'] = flux_high
            #phottable['flux_low'] = flux_low
            phottable['AIRMASS'] = header['AIRMASS']
            for col in phottable.colnames:
                phottable[col].info.format = '%.4f'           
            #phottable['aveFWHM'] = np.round(fwhm,2)
            #phottable['JD-HELIO'] = header['JD-HELIO']
            phottable['MJD-HELIO'] = header['JD-HELIO']-2400000.5
            phottable['FILTER'] = header['FILTER']
            phottable['DATE-OBS'] = header['DATE-OBS']     
            phottable['flag'] = flag
            #phottable['ID'] = sources['id']
            # Output table name will include FWHM used for aperture photometry
            phottable.write(fl.replace("fit","txt").replace("astrometry","photometry"), format='ascii', overwrite=True)
            photfile = prefix+'_photometry.txt'
            if not (os.path.exists(photfile) and os.path.getsize(photfile) > 0):
                 headline = ['FILENAME','MJD-HELIO','FILTER','DATE-OBS', 'RAJ2000','DECJ2000','xpix','ypix','flag','flux_peak','flux','mag','mag_err']
                 headline.append('\n')
            else:
                 headline=[]
           # UBACI RA,DEC u ovaj fajl dole
            catalog = SkyCoord(ra=world[:,0]*u.degree, dec=world[:,1]*u.degree)
            targets=np.genfromtxt('katalog.cat', delimiter=' ', dtype=None, 
                names=['name','ra','dec'], encoding=None)
            target_idx = [n for n, x in enumerate(targets['name']) if object_name in x]
            object_ra, object_dec = targets['ra'][target_idx][0], targets['dec'][target_idx][0]
            object_skycoo = SkyCoord(ra=object_ra, dec=object_dec, unit=(u.hourangle, u.deg))
            idx, d2d, d3d = object_skycoo.match_to_catalog_sky(catalog)
            with open(prefix+'_photometry.txt', mode='a+') as outname:
                magnitude=[flux_peak[idx], flux[idx], mag[idx], mag_err[idx]]
                txtline=[]                 
                txtline=[fl,str(header['JD-HELIO']-2400000.5),str(header['FILTER']),str(header['DATE-OBS']),world[:,0][idx], world[:,1][idx], positions[idx][0], positions[idx][1],flag[idx]]
                outline=headline+txtline+magnitude
                outline.append('\n')
                outname.writelines("%s " % item for item in outline)
                outname.close()
      # plot
      if ifPlotPhotApertures:
            interval = ZScaleInterval()
            zmin, zmax = interval.get_limits(data)
            fig, ax = pl.subplots(1,1, figsize=(12,12))
            ax.set_title(os.path.basename(fl))
            ax.imshow(data, origin='lower', cmap='Greys_r', vmin=zmin, vmax=zmax, interpolation='nearest')
            if os.path.exists(targets_list) and os.path.getsize(targets_list) > 0:
                 idx = 0
            for i in range(len(positions)):
                if i == idx:
                    circle = pl.Circle((positions[idx,0], positions[idx,1]), radius=15, color='blue', lw=2, alpha=.7, fill=False)
                else:
                    circle = pl.Circle((positions[i,0], positions[i,1]), radius=25, color='yellow', lw=1, alpha=.5, fill=False)
                ax.add_patch(circle)
            pl.savefig(fl.replace("fit", "png").replace("astrometry","photometry"), format='png')


# def photometry_multiprocess(FITSfls_list, targets_list):

#       # use more threads
#       with concurrent.futures.ThreadPoolExecutor() as executor:
#         args = ((fl, targets_list) for fl in FITSfls_list)
#         executor.map(doPhotometry, args)

def photometry_multiprocess(FITSfls_list, prefix):

      # use more threads
      n_processes = mp.cpu_count() - 1 
      with concurrent.futures.ThreadPoolExecutor(max_workers=n_processes) as executor:
        args = ((fl, prefix) for fl in FITSfls_list)
        executor.map(doPhotometry, args)
        #executor.map(doCheck, args)

##################################### MAIN ############################## 
# Prerequisites:
# RAM > 16GB for sky flat
# solve-filed: Astrometry 0.89
# Python 3.9.10
# numpy.version > 1.22.2
# anaconda - pozeljno
# astropy.version > 5.2.1
# photutils.version == 1.5.0
# conda-forge ccdproc
import sys
import os
import glob
#import fnmatch
import time

from astropy.io import fits
import numpy as np
#import matplotlib.pyplot as plt
#from astropy.visualization import simple_norm
#from photutils.segmentation import make_source_mask
from astropy.stats import sigma_clipped_stats
#import numpy.ma as ma
from astropy.wcs import WCS
from astropy import units as u
#import subprocess
import csv
#from astropy.convolution import Gaussian2DKernel
#from astropy.convolution import convolve

from photutils.detection import DAOStarFinder, IRAFStarFinder
#from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import ApertureStats
from astropy.stats import SigmaClip

#from astropy.modeling.models import Gaussian2D
from astropy.modeling import models, fitting
#from astropy.io import ascii
from astropy.table import Table
from astropy.nddata.utils import Cutout2D

import warnings
import logging
logging.disable(logging.INFO)
warnings.simplefilter('ignore')

# naknadno dodati moduli i funkcije
import ccdproc as ccdp
import multiprocessing as mp
import concurrent.futures
from scipy.ndimage import median_filter
from astropy.nddata import CCDData
from scipy import stats, interpolate, ndimage
from astropy.visualization import ZScaleInterval
from astropy.coordinates import SkyCoord
import matplotlib.pylab as pl
#from matplotlib.patches import Circle
import psutil
import math

# NEW -------
from photutils.segmentation import detect_sources
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import (detect_sources, make_2dgaussian_kernel)
from photutils.aperture import CircularAperture
from photutils.utils import circular_footprint 
from photutils.centroids import centroid_sources,  centroid_2dg
from photutils.utils import circular_footprint
import pandas as pd

import twirl
import concurrent.futures
import argparse

from astropy.wcs.utils import proj_plane_pixel_scales

#from scipy.odr import *

#from scipy.optimize import curve_fit
from astropy.coordinates import match_coordinates_sky
from astroquery.gaia import Gaia
from lmfit.models import LinearModel

start_time = time.time()
log_time = open('log_time.dat', 'a')

fit_files = "*.fit"
# Change the path to the raw data folder where the pipeline should be put
# path = "/home/milan/test_pipeline/12_09_2023_iraf/"
# os.chdir(path)
def create_arg_parser():
    # Creates and returns the ArgumentParser object

    parser = argparse.ArgumentParser(description='Description of your app.')
    parser.add_argument('work_dir',
                    help='Path to the working directory containing raw data.')
    return parser

if __name__ == "__main__":
    
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    #print(parsed_args)
    if os.path.exists(parsed_args.work_dir):
        path = parsed_args.work_dir
        os.chdir(path)
    else:
        raise ValueError('Enter full path to the existing dir.')

    # path = "/home/milan/test_pipeline/13_10_2023_iraf/"
    # os.chdir(path)
    # print(path)

    if len(glob.glob(fit_files)) > 0: 
        ifc = ccdp.ImageFileCollection(path, filenames=glob.glob(fit_files))
    else: 
        raise ValueError("There are no .fit files in this working dir.")
    # make a list of BIAS images
    biasNms_list = ifc.files_filtered(imagetyp='bias')
    if len(biasNms_list)>0:
        print("Number of bias frames: ", len(biasNms_list))
    else:
        raise ValueError("There are no BIAS .fit files in this working dir.")
    # dictionary: all DARK frames with key=EXPTIME and values=list of DARK frames
    darkNms_dic = {}
    ifc_dark = ifc.filter(regex_match=True, imagetyp='dark')
    ndarks = len(ifc.files_filtered(imagetyp='dark'))
    if ndarks>0:
        print("Number of dark frames: ", ndarks)
    else:
        raise ValueError("There are no DARK .fit files in this working dir.")
    ifc_dark_groups = ifc_dark.summary.group_by('exptime')
    for k,g in zip(ifc_dark_groups.groups.keys, ifc_dark_groups.groups):
      darkNms_dic[float(k['exptime'])] = [os.path.basename(fl) for fl in list(g['file'])]
    
    # dictionary: all FLAT frames with key=FILTER and values=list of FLAT frames
    flatNms_dic = {}
    ifc_flat = ifc.filter(regex_match=True, imagetyp='flat')
    nflats = len(ifc.files_filtered(imagetyp='flat'))
    if nflats>0:
        print("Number of flat frames: ", nflats)
    else:
        raise ValueError("There are no FLAT .fit files in this working dir.")
    
    ifc_flat_groups = ifc_flat.summary.group_by('filter')
    for k,g in zip(ifc_flat_groups.groups.keys, ifc_flat_groups.groups):
      flatNms_dic[str(k['filter'])] = [os.path.basename(fl) for fl in list(g['file'])] 
      
    # list of LIGHT frames
    lightNms_list = ifc.files_filtered(imagetyp='light')
    if len(lightNms_list) == 0:
        raise ValueError("There are no LIGHT .fit files (science images) in this working dir.")
    num_of_objects = np.unique(list(map(lambda x: x.split('_')[0].split('-')[0],lightNms_list))).size
    object_names = np.unique(list(map(lambda x: x.split('_')[0].split('-')[0],lightNms_list)))


    global calibFolderNm, astroFolderNm, photoFolderNm, ifPauseProcess, ifPlotPhotApertures, ifFWHMvariable, ifPlotCalibratedFrames, ifBadPixels
    calibFolderNm = "./calibrated"
    astroFolderNm = "./astrometry"
    photoFolderNm = "./photometry"
    ifPauseProcess = True          # pause processes while doAstrometry() for CPU
    ifPlotPhotApertures = False     # imshow() aperture for visual check (True); or save jpg (False)
    ifFWHMvariable = False
    ifPlotCalibratedFrames = False
    ifBadPixels=False
    
    inputFls_list = glob.glob(os.path.join(astroFolderNm, "*cal_wcs.fit"))
    outputFls_list = glob.glob(os.path.join(photoFolderNm, "*cal_wcs.txt"))
    #print(inputFls_list)

    # ----- make new folders to store output data: calibrated ([calibrated]) images, 
    # plate solved ([astrometry]) images with aperture photometry done ([photometry])
    
    if not os.path.isdir(os.path.join(path,calibFolderNm)):
      os.makedirs(os.path.join(path,calibFolderNm))
    if not os.path.isdir(os.path.join(path,astroFolderNm)):
      os.makedirs(os.path.join(path,astroFolderNm))  
    if not os.path.isdir(os.path.join(path,photoFolderNm)):
      os.makedirs(os.path.join(path,photoFolderNm))  
    
    
    
    # ------ gather all FITS images
    
    ifc = ccdp.ImageFileCollection(path, filenames=glob.glob(fit_files))
    
    # make a list of BIAS images
    biasNms_list = ifc.files_filtered(imagetyp='bias')
    
    # dictionary: all DARK frames with key=EXPTIME and values=list of DARK frames
    darkNms_dic = {}
    ifc_dark = ifc.filter(regex_match=True, imagetyp='dark')
    ifc_dark_groups = ifc_dark.summary.group_by('exptime')
    for k,g in zip(ifc_dark_groups.groups.keys, ifc_dark_groups.groups):
      darkNms_dic[float(k['exptime'])] = [os.path.basename(fl) for fl in list(g['file'])]
    
    # dictionary: all FLAT frames with key=FILTER and values=list of FLAT frames
    flatNms_dic = {}
    ifc_flat = ifc.filter(regex_match=True, imagetyp='flat')
    ifc_flat_groups = ifc_flat.summary.group_by('filter')
    for k,g in zip(ifc_flat_groups.groups.keys, ifc_flat_groups.groups):
      flatNms_dic[str(k['filter'])] = [os.path.basename(fl) for fl in list(g['file'])] 
    
    # dictionary: all LIGHT frames with key=FILTER and values=list of LIGHT frames
    skyflatNms_dic = {}
    ifc_light = ifc.filter(regex_match=True, imagetyp='light')
    ifc_light_groups = ifc_light.summary.group_by('filter')
    for k,g in zip(ifc_light_groups.groups.keys, ifc_light_groups.groups):
      skyflatNms_dic[str(k['filter'])] = [os.path.basename(fl) for fl in list(g['file'])] 
    print(skyflatNms_dic.items())  
    # list of LIGHT frames
    lightNms_list = ifc.files_filtered(imagetyp='light')
    
    
    
    # ------ make master BIAS
    
    if not os.path.isfile(os.path.join(calibFolderNm, "mBias.fits")):
      # make master bias
      mBias = makeMasterBIAS(biasNms_list)
      print("Making master bias: mBias.fits\n")
      
      # SAVE
      primHDU = fits.PrimaryHDU()
      primHDU.data = np.float32(mBias)
      primHDU.header['HISTORY'] = "Master bias frame"
      hdulist = fits.HDUList([primHDU])
      hdulist.writeto(os.path.join(calibFolderNm, "mBias.fits"), overwrite=True)
    
    else:
      # load
      with fits.open(os.path.join(calibFolderNm, "mBias.fits")) as hdul:
        mBias = hdul[0].data
      print("Reading master bias: mBias.fits\n")
    
    
    
    # ----- make master DARK frames as dictionary entries
    
    # output: mDark_dic = {5:np.array, 200:np.array, 600:np.array ... }
    mDark_dic = {}
    for expt,darkNms_list in darkNms_dic.items():
      
      # create name for master dark frame
      dark_outputNm = "mDark_{}sec.fits".format(str(int(expt)))
      
      if not os.path.isfile(os.path.join(calibFolderNm, dark_outputNm)):
        
        # make master dark
        mDark = makeMasterDARK(darkNms_list, mBias)
        print(f'Making master dark: {dark_outputNm}\n')
        
        # SAVE
        primHDU = fits.PrimaryHDU()
        primHDU.data = np.float32(mDark)
        primHDU.header['HISTORY'] = "Master dark frame"
        hdulist = fits.HDUList([primHDU])
        hdulist.writeto(os.path.join(calibFolderNm, dark_outputNm), overwrite=True)
        
        # put into a dictionary
        mDark_dic[expt] = mDark
        
      else:
        
        # load
        with fits.open(os.path.join(calibFolderNm, dark_outputNm)) as hdul:
          mDark = hdul[0].data
        print(f'Reading master dark: {dark_outputNm}\n')
          
        # put into a dictionary
        mDark_dic[expt] = mDark
      
    
    # ----- make master FLAT frames as dictionary entries
    
    # output: mFlat_dic = {'B':np.array, 'V':np.array, ... }
    mFlat_dic = {}
    for filt,flatNms_list in flatNms_dic.items():
      
      # konstruisi ime za master dark frame
      flat_outputNm = "mFlat_{}.fits".format(filt)
      
      if not os.path.isfile(os.path.join(calibFolderNm, flat_outputNm)):
        
        # napravi master dark
        mFlat = makeMasterFLAT(flatNms_list, mBias, mDark_dic)
        print(f'Making master flat: {flat_outputNm}\n')
        
        # snimi
        primHDU = fits.PrimaryHDU()
        primHDU.data = np.float32(mFlat)
        primHDU.header['HISTORY'] = "Master flat frame"
        hdulist = fits.HDUList([primHDU])
        hdulist.writeto(os.path.join(calibFolderNm, flat_outputNm), overwrite=True)
        
        # put into a dictionary
        mFlat_dic[filt] = mFlat
        
      else:
        
        # load
        with fits.open(os.path.join(calibFolderNm, flat_outputNm)) as hdul:
          mFlat = hdul[0].data
        print(f'Reading master flat: {flat_outputNm}\n')
          
        # put into a dictionary
        mFlat_dic[filt] = mFlat
      
    
    
    
    # ----- make master skyflats as dictionary entries
    
    # output: mSkyflat_dic = {'B':np.array, 'V':np.array, None, ... }
    mSkyflat_dic = {}
    for filt,skyflatNms_list in skyflatNms_dic.items():
      # konstruisi ime za master dark frame
      skyflat_outputNm = "mSkyflat_{}.fits".format(filt)
      if not os.path.isfile(os.path.join(calibFolderNm, skyflat_outputNm)):
        
        # make master dark
        mSkyflat = makeMasterSKYFLAT_multiprocess(skyflatNms_list, mBias, mDark_dic, mFlat_dic)
        #mSkyflat = makeMasterSKYFLAT_singleprocess(skyflatNms_list, mBias, mDark_dic, mFlat_dic)
        # SAVE
        if mSkyflat is not None:
          primHDU = fits.PrimaryHDU()
          primHDU.data = np.float32(mSkyflat)
          primHDU.header['HISTORY'] = f"Master sky-flat frame in {filt} filter"
          hdulist = fits.HDUList([primHDU])
          hdulist.writeto(os.path.join(calibFolderNm, skyflat_outputNm), overwrite=True)
          print(f'Making master sky flat: {skyflat_outputNm}\n')
        
        # put into a dictionary
        mSkyflat_dic[filt] = mSkyflat
        
      else:
    
        # load
        with fits.open(os.path.join(calibFolderNm, skyflat_outputNm)) as hdul:
          mSkyflat = hdul[0].data
          
        # put into a dictionary
        mSkyflat_dic[filt] = mSkyflat
        
        #informer
        if mSkyflat_dic[filt] is not None:
          print(f'Reading master sky flat: {skyflat_outputNm}\n')
        
    
    # ----- Calibrate LIGHT frames
    
    inputFls_list = glob.glob(os.path.join(calibFolderNm, "*_cal.fit")) 
    print(len(inputFls_list), len(lightNms_list))
    if len(inputFls_list) < len(lightNms_list):
      # kalibrisi
      calibrate_multiprocess(lightNms_list, mBias, mDark_dic, mFlat_dic, mSkyflat_dic)
    else:
      print('Calibration of LIGHT frames is already done.\n')
    
    def fixpix(lightNms_list, darkNms_list, hotPixThresh=10.):
      
      # napravi prvo hot/dead pixel masku
      if not os.path.isfile("hotPixMask.fits"):
      
        # median scDarks
        frame_list = []
        for fl in darkNms_list:
          with fits.open(fl) as hdul:
            frame_list.append(hdul[0].data)
        frame_arr = np.array(frame_list)
        medDark = np.median(frame_arr, axis=0)
        
        
        # izracunaj hotPixThresh u odnosu na vrednosti piksela
        medBoxSz = 5
        mDark_blurred = median_filter(medDark, medBoxSz)  # scipy function
        difference = medDark - mDark_blurred
        hotPixThresh = hotPixThresh * np.std(difference)
        
        # napravi hot pixel mask 
        hotPixMask = np.zeros(medDark.shape)
        hotPixMask[(np.abs(difference)>hotPixThresh)] = 1
        
         # ispisi kao uint
        hotPixMask_ccd = CCDData(hotPixMask.astype('uint8'), unit=u.dimensionless_unscaled)
        header = fits.Header()
        header['imagetyp'] = "HOTPIXMASK"
        header['comment'] = "number of hot pixels: {}".format(int(hotPixMask.sum()))
        hotPixMask_ccd.header = header
        hotPixMask_ccd.write(os.path.join(calibFolderNm, "hotPixMask.fits"), overwrite=True)
      
      else:
        
        with fits.open(os.path.join(calibFolderNm, "hotPixMask.fits")) as hdul:
          hotPixMask = hdul[0].data  
          
      
      # za svaku sliku pravi CR  masku i kombinuj za hot/deadmaskom
      for fl in lightNms_list:
        
        print(f"\n-----> fixpix {os.path.basename(fl)}")
       
        with fits.open(fl) as hdul:
          data = hdul[0].data
          header = hdul[0].header
        
        gain = 1. # e/ADU
        _, cosRayMask = ccdp.cosmicray_lacosmic(data, sigclip=10, gain=gain, niter=3)
        
        # info
        print(f"    Numer of hot/dead pixels: {int(np.sum(hotPixMask))}")
        print(f"    Numer of cosmic rays: {int(np.sum(cosRayMask))}")
        
        finalMask = np.logical_or(hotPixMask.astype(bool), cosRayMask.astype(bool))
        finalmask_ccd = CCDData(cosRayMask.astype('uint8'), unit=u.dimensionless_unscaled)
        finalmask_ccd.write(fl.replace(".fit", "_CRmask.fit"), overwrite=True)
      
        # korekcija
        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        
        # meshgrid
        xx, yy = np.meshgrid(x, y) # xx (2048, 2048), yy (2048, 2048)
        
        # masked data
        mdata = np.ma.masked_array(data, mask=finalMask==1) 
        
        # pozicije losih vrednosti koje treba da odredis interpolacijom
        missing_x = xx[mdata.mask]  # (1291,) x indeksi maskiranih piksela
        missing_y = yy[mdata.mask]  # (1291,) y indeksi maskiranih piksela
        
        # interpolacija
        interp_values = ndimage.map_coordinates(mdata, [missing_x, missing_y], order=1)
        
        # fiksiraj piksele
        interp_mdata = mdata.copy()
        interp_mdata[missing_y, missing_x] = interp_values  # vodi racuna da prvi ide missing_y, pa missing_x!!!
        
        # snimi
        interp_mdata_ccd = CCDData(interp_mdata, unit='adu')/header['EXPTIME']
        interp_mdata_ccd.header = header
        interp_mdata_ccd.header['FIXPIX'] = (True,"Hot pixels fixed")
        interp_mdata_ccd.write(fl.replace(".fit", "_fix.fit"), overwrite=True)
    
    
    
    # ------- Remove bad pixels 
    if ifBadPixels:
    
        inputFls_list = glob.glob(os.path.join(calibFolderNm, "*_cal.fit"))
        outputFls_list = glob.glob(os.path.join(calibFolderNm, "*_fix.fit"))
    
        if len(outputFls_list) == 0:
      
            maxDarkNms_list = darkNms_dic[max(darkNms_dic)]
      
            # correct for cosmic rays and hot/dead pixels
            fixpix(inputFls_list, maxDarkNms_list, hotPixThresh=10.)
      
            # copy *fix.fit files into ./astrometry folder
            os.system("cp {} {}".format(os.path.join(calibFolderNm, '*fix.fit'), astroFolderNm))
      
        else:
            print('Cosmic rays and hot/dead pixels already removed\n')
    
    # --------- Do astrometry
    
    if ifBadPixels:
        inputFls_list = glob.glob(os.path.join(calibFolderNm, "*fix.fit"))
        outputFls_list = glob.glob(os.path.join(astroFolderNm, "*fix_wcs.fit"))
    else:
        inputFls_list = glob.glob(os.path.join(calibFolderNm, "*cal.fit"))
        outputFls_list = glob.glob(os.path.join(astroFolderNm, "*cal_wcs.fit"))
    #print("Input file list for  astrometry")
    print("Input file list for  astrometry")
    print(inputFls_list)
    if len(outputFls_list) < len(inputFls_list):
      print("start astrometry:")
      tmp = list(map(lambda x: x.replace('_wcs','').replace('plate_solved','calibrated'),outputFls_list))
      inputFls_list_new =  [x for x in inputFls_list if x not in set(tmp)]
      doAstrometry(inputFls_list_new)
      print("astrometry done")
    else:
      print('Astrometry already done.\n')

# Photometry
    if ifBadPixels:
        inputFls_list = glob.glob(os.path.join(astroFolderNm, "*fix.fit"))
        outputFls_list = glob.glob(os.path.join(photoFolderNm, "*fix_wcs.txt"))
    else:
        inputFls_list = glob.glob(os.path.join(astroFolderNm, "*cal_wcs.fit"))
        outputFls_list = glob.glob(os.path.join(photoFolderNm, "*cal_wcs.txt"))

    inputFls_list_new = []
    for obj_num in range(0,num_of_objects):
        
        print('Object::', object_names[obj_num])
        prefix = object_names[obj_num]
        targets_list = object_names[obj_num] + '_stars.txt'
        phot_list = object_names[obj_num] + '_photometry.txt'
        match_input = [s for s in inputFls_list if object_names[obj_num] in s]
        print('-------------------')
        if (os.path.exists(targets_list)): #    and (os.path.getsize(targets_list) > 0:
            #match_output_size = sum(1 for _ in open(phot_list)) - 2 # -2: 1 for header and 1 for last empty line
            if os.path.exists(phot_list) and os.path.getsize(phot_list) > 0:
                done = np.loadtxt(phot_list, usecols=(0,),dtype=str).tolist()
                print('Number of input frames:', len(match_input))
                print("Already done for", len(done)-1, ' input frames.')
                if len(match_input) > len(done)-1:  
                    inputFls_list_new = [x for x in match_input if x not in done]
                    print('No. of missing files:', len(inputFls_list_new))
#                    print(inputFls_list_new)
                else:
                    print('Photometry on ' + object_names[obj_num] + ' is done.\n')
            else:
                inputFls_list_new = match_input
                
        else:
            match_output = [s for s in outputFls_list if object_names[obj_num] in s]
            if len(match_output) < len(match_input):
                tmp = list(map(lambda x: x.replace('txt','fit').replace('photometry','astrometry'),match_output))
                inputFls_list_new =  [x for x in match_input if x not in set(tmp)]
                print('No. of missing files:', len(match_input),' (input) -',len(match_output),'(done) =',len(inputFls_list_new))
                #print(os.path.exists(targets_list) and os.path.getsize(targets_list) > 0)
                #print(inputFls_list_new)
            else:
                print('Photometry on ' + object_names[obj_num] + ' is already done.\n')
        # print("Input files:", len(match_input))
        # print("Output files:", len(match_output))
        if len(inputFls_list_new) > 0:   
            #print(object_names[obj_num]," phot list: ", inputFls_list_new)
            photometry_multiprocess(inputFls_list_new, prefix)
        
    #print("Ne pravi Mrk335_photometry.txt fajl.")
    round(time.time() - start_time, 2)
    print ("---------- %s seconds ----------" % round(time.time() - start_time, 4), "END")
    print ("---------- %s seconds ----------" % round(time.time() - start_time, 4), file=log_time)
    log_time.close()



# --------- Do aperture photometry
# Filename of the file that lists target (1st line) and comparions stars: RAJ2000 DECJ2000 id
# If such file is not provided, all sources in each frame will be selected and processed
#targets_list='Mrk335_comp_stars.txt'
# If comparison stars are not provided(targets_list is an empty file) than
# ifFWHMvariable should be set to False, otherwise True
#ifFWHMvariable = False
# Count number of objects imaged for the night and make as many arrays to keep 
# values of magnitudes, and extra info on JD, AIRMASS etc.

