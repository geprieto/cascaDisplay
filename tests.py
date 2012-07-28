import glob
import pyfits
import numpy
from numpy import zeros
import numpy.ma as ma
import scipy.stats as st
from math import atan2, cos, sin, radians, degrees
from pyraf import iraf
import img_scale
import os
import pylab as py
import matplotlib.pyplot as plt

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
	header = getHeader(fits)
	header.update('BIASSUB', 'YES')
	pyfits.writeto(newfits, newdata, header)

#doBiasSubtraction('lc_b20120520ut041236s72330.fits', 'fits/BIAS.fits', 'testunbias.fits')
	
def doDarkSubtraction(fits,dark,newfits):
	header = getHeader(fits)
	exptime = float(header['EXPTIME'])
	correctedDarkData = getData(dark) * exptime
	fitsData = getData(fits)
	newdata = fitsData - correctedDarkData
	header.update('DARKSUB', 'YES')
	pyfits.writeto(newfits, newdata, header)

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

def getStats(fits, mask):
	fitsData = getData(fits)
	maskData = getData(mask)
	fitsHeader = getHeader(fits)
	dataMasked = ma.array(fitsData, mask=maskData)
	exptime = float(fitsHeader['EXPTIME'])
	dataScaled = dataMasked / exptime
	mean = numpy.mean(dataScaled)
	average = numpy.average(dataScaled)
	amin = numpy.amin(dataScaled)
	amax = numpy.amax(dataScaled)
	median = numpy.median(dataScaled)
	std = numpy.std(dataScaled)
	var = numpy.var(dataScaled)
	return {'mean' : mean, 'average' : average, 'min' : amin, \
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
	
def createImage(fits,imagefile):
	os.system("rm temp*.fits")
	doBiasSubtraction(fits,'fits/BIAS.fits','tempunbias.fits')
	doDarkSubtraction('tempunbias.fits','fits/DARK.fits','tempundark.fits')
	maskMoon(fits,'fits/MASK.fits','tempmoonmask.fits')
	stats = getStats('tempundark.fits','tempmoonmask.fits')
	min = stats['min']
	max = stats['max']
	mean = stats['mean']
	std = stats['std']
	pixMin = (min + mean - std)/2
	pixMax = (max + mean + std)/2
	data = getData('tempundark.fits')
	x, y = data.shape
	img = img_scale.sqrt(data, min, pixMax)
	py.imshow(img, aspect='equal', cmap=plt.get_cmap('gray'))
	py.savefig(imagefile,dpi=300)

# full moon example	
createImage('lc_r20120604ut041627s76110.fits','fullmoon.png')

# new moon example
createImage('lc_r20120520ut041206s72300.fits','newmoon.png')

# 10% clouds, no moon example
createImage('lc_r20120615ut072304s03540.fits','10clouds0moon.png')

# 70% clouds moon up example
createImage('lc_r20120609ut071844s01860.fits','70clouds50moon.png')
	

	


	

