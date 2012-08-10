import img_scale
import fitsimage

import glob
import pyfits
import numpy
from numpy import zeros
import numpy.ma as ma
import scipy.stats as st
from math import atan2, cos, sin, radians, degrees
from pyraf import iraf
import os
import pylab as py
import matplotlib.pyplot as plt
import pointarray
import Image

XMIN, XMAX = 33.0, 985.0
YMIN, YMAX = 24.0, 982.0
NORTHX, NORTHY = 472.0, 17.0

def getLastFile(directory):
	filepath = directory + '/*.fits'
	filesList = glob.glob(filepath)
	return filesList[-1]
	
def getHeader(fitsFile):
	fitsImage = pyfits.open(fitsFile)
	fitsHeader = fitsImage[0].header
	fitsImage.close()
	return fitsHeader
	
def getData(fitsFile):
	fitsImage = pyfits.open(fitsFile)
	fitsData = fitsImage[0].data
	fitsImage.close()
	return fitsData
	
def doBiasSubtraction(fits,bias,newfits):
	data1 = getData(fits)
	data2 = getData(bias)
	newdata = data1 - data2
	#header = getHeader(fits)
	#header.update('BIASSUB', 'YES')
	#pyfits.writeto(newfits, newdata, header)
	return newdata

#doBiasSubtraction('lc_b20120520ut041236s72330.fits', 'fits/BIAS.fits', 'testunbias.fits')
	
def doDarkSubtraction(fits,dark,bias,newfits):
	header = getHeader(fits)
	exptime = float(header['EXPTIME'])
	correctedDarkData = getData(dark) * exptime
	#fitsData = getData(fits)
	fitsData = doBiasSubtraction(fits,bias,newfits)
	newdata = fitsData - correctedDarkData
	#header.update('DARKSUB', 'YES')
	#pyfits.writeto(newfits, newdata, header)
	return newdata

#doDarkSubtraction('testunbias.fits','fits/DARK.fits','testundark.fits')

def createMask(fits,newMask,terrainCounts,saturationCounts):
	data = getData(fits)
	dataCopy = data.copy
	maskTerrain = data < terrainCounts
	data[maskTerrain] = 1
	unmask = data >= terrainCounts
	data[unmask] = 0
	lx, ly = data.shape
	X, Y = numpy.ogrid[0:lx, 0:ly]
	maskBirdShit = (X - lx/2)**2 + (Y - ly/2)**2 < lx*ly/6
	data[maskBirdShit] = 0
	maskSaturation = dataCopy > saturationCounts
	data[maskSaturation] = 1
	pyfits.writeto(newMask, data)
	
#createMask('testundark.fits', 'testmask.fits', 300.0, 1000.0)

def getStats(fitsData, mask):
	#fitsData = getData(fits)
	maskData = getData(mask)
	#fitsHeader = getHeader(fits)
	dataMasked = ma.array(fitsData, mask=maskData)
	mean = numpy.mean(dataMasked)
	amin = numpy.amin(dataMasked)
	amax = numpy.amax(dataMasked)
	median = numpy.median(dataMasked)
	std = numpy.std(dataMasked)
	var = numpy.var(dataMasked)
	return {'mean' : mean, 'min' : amin, \
	        'max' : amax, 'median' : median, 'std' : std, \
	        'var' : var}	
#print getStats('testundark.fits','fits/MASK.fits')

def getAltCorrection(objectEl, radii):
	linearCorrection = 1.0
	x = radii * cos(radians(objectEl)) * linearCorrection
	return x
	
def getCoords(objectEl, objectAz):
	xRadii = (XMAX - XMIN)/2
	yRadii = (YMAX - YMIN)/2
	radii = (xRadii + yRadii)/2
	centerX = XMIN + radii
	centerY = YMIN + radii
	northXVector = NORTHX - centerX
	northYVector = NORTHY - centerY
	angleFromNorth = atan2(northYVector,northXVector)
	objectRadii = getAltCorrection(objectEl, radii)
	objectTheta = angleFromNorth - radians(objectAz)
	objectX = objectRadii * cos(objectTheta) + centerX
	objectY = objectRadii * sin(objectTheta) + centerY
	return objectX, objectY
	
def addToMask(mask,newMask,xCoord,yCoord,radii):
	data = getData(mask)
	lx, ly = data.shape
	X, Y = numpy.ogrid[0:lx, 0:ly]
	maskRegion = (X - yCoord)**2 + (Y - xCoord)**2 < radii**2
	data[maskRegion] = 1
	pyfits.writeto(newMask, data)

def maskMoon(fits, mask, moonMask):
	header = getHeader(fits)
	moonAz = iraf.real(header['MOONAZ'])
	moonEl = iraf.real(header['MOONEL'])
	if (moonEl > 0):
		moonX, moonY = getCoords(moonEl, moonAz)
		addToMask(mask, moonMask, moonX, moonY, 200)
	else:
		os.system("cp "+mask+" "+moonMask)
	
def zscale_range(image_data_unmasked, contrast=0.25, num_points=600, num_per_row=120, mask=None):

    if (mask==None):
        image_data = image_data_unmasked
    else:
        image_data = ma.array(image_data_unmasked, mask=mask)

    # check contrast
    if contrast <= 0.0:
        contrast = 1.0

    # check number of points to use is sane
    if num_points > numpy.size(image_data) or num_points < 0:
        num_points = 0.5 * numpy.size(image_data)

    # determine the number of points in each column
    num_per_col = int(float(num_points) / float(num_per_row) + 0.5)

    # integers that determine how to sample the control points
    xsize, ysize = image_data.shape
    row_skip = float(xsize - 1) / float(num_per_row - 1)
    col_skip = float(ysize - 1) / float(num_per_col - 1)

    # create a regular subsampled grid which includes the corners and edges,
    # indexing from 0 to xsize - 1, ysize - 1
    data = []
   
    for i in xrange(num_per_row):
        x = int(i * row_skip + 0.5)
        for j in xrange(num_per_col):
            y = int(j * col_skip + 0.5)
            if image_data[x, y] > 0:
            	data.append(image_data[x, y])

    # actual number of points selected
    num_pixels = len(data)

    # sort the data by intensity
    data.sort()

    # check for a flat distribution of pixels
    data_min = min(data)
    data_max = max(data)
    center_pixel = (num_pixels + 1) / 2
    
    if data_min == data_max:
        return data_min, data_max

    # compute the median
    if num_pixels % 2 == 0:
        median = data[center_pixel - 1]
    else:
        median = 0.5 * (data[center_pixel - 1] + data[center_pixel])

    # compute an iterative fit to intensity
    pixel_indeces = map(float, xrange(num_pixels))
    points = pointarray.PointArray(pixel_indeces, data, min_err=1.0e-4)
    fit = points.sigmaIterate()

    num_allowed = 0
    for pt in points.allowedPoints():
        num_allowed += 1

    if num_allowed < int(num_pixels / 2.0):
        return data_min, data_max

    # compute the limits
    z1 = median - (center_pixel - 1) * (fit.slope / contrast)
    z2 = median + (num_pixels - center_pixel) * (fit.slope / contrast)

    if z1 > data_min:
        zmin = z1
    else:
        zmin = data_min

    if z2 < data_max:
        zmax = z2
    else:
        zmax = data_max

    # last ditch sanity check
    if zmin >= zmax:
        zmin = data_min
        zmax = data_max

    return zmin, zmax
    
def FitsImage(fitsfile, data, contrast_opts={}, scale="linear",
              scale_opts={}, mask=None):

    # open the fits file and read the image data and size
    #fitslib.fits_simple_verify(fitsfile)
    fits = pyfits.open(fitsfile)

    try:
        hdr = fits[0].header
        xsize = hdr["NAXIS1"]
        ysize = hdr["NAXIS2"]
        fits_data = data
    finally:
        fits.close()
        
    if (mask!=None):
    	maskfits_data = getData(mask)
    
    # compute the proper scaling for the image
    contrast_value = contrast_opts.get("contrast", 0.25)
    num_points = contrast_opts.get("num_points", 600)
    num_per_row = contrast_opts.get("num_per_row", 120)
    zmin, zmax = zscale_range(fits_data, contrast=contrast_value,
                              num_points=num_points,
                              num_per_row=num_per_row, mask=maskfits_data)

    # set all points less than zmin to zmin and points greater than
    # zmax to zmax
    fits_data = numpy.where(fits_data > zmin, fits_data, zmin)
    fits_data = numpy.where(fits_data < zmax, fits_data, zmax)

    if scale == "linear":
        scaled_data = (fits_data - zmin) * (255.0 / (zmax - zmin)) + 0.5
    elif scale == "arcsinh":
        # nonlinearity sets the range over which we sample values of the
        # asinh function; values near 0 are linear and values near infinity
        # are logarithmic
        nonlinearity = scale_opts.get("nonlinearity", 3.0)
        nonlinearity = max(nonlinearity, 0.001)
        max_asinh = cmath.asinh(nonlinearity).real
        scaled_data = (255.0 / max_asinh) * \
                      (numpy.arcsinh((fits_data - zmin) * \
                                     (nonlinearity / (zmax - zmin))))

    # convert to 8 bit unsigned int ("b" in numpy)
    scaled_data = scaled_data.astype("b")
    
    # create the image
    image = Image.frombuffer("L", (xsize, ysize), scaled_data)
    return image
    
def createImage(fits,imagefile):
	os.system("rm temp*.fits")
	#doBiasSubtraction(fits,'fits/BIAS.fits','tempunbias.fits')
	data = doDarkSubtraction(fits,'fits/DARK.fits','fits/BIAS.fits','tempundark.fits')
	maskMoon(fits,'fits/MASK.fits','tempmoonmask.fits')
	#stats = getStats('tempundark.fits','tempmoonmask.fits')
	#min = stats['min']
	#max = stats['max']
	#mean = stats['mean']
	#std = stats['std']
	#pixMin = (min + mean - std)/2
	#pixMax = (max + mean + std)/2
	img = FitsImage(fits, data, scale="linear", mask='tempmoonmask.fits')
	py.imshow(img, aspect='equal', cmap=plt.get_cmap('gray'))
	py.savefig(imagefile,dpi=300)
	os.system("rm temp*.fits")

# full moon example	
createImage('lc_r20120604ut041627s76110.fits','fullmoon.png')

# new moon example
#createImage('lc_r20120520ut041206s72300.fits','newmoon.png')

# 10% clouds, no moon example
#createImage('lc_r20120615ut072304s03540.fits','10clouds0moon.png')

# 70% clouds moon up example
#createImage('lc_r20120609ut071844s01860.fits','70clouds50moon.png')


	


	

