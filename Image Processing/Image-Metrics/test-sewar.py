#-*- coding:utf-8 -*-
import time
import os
import multiprocessing 
from multiprocessing import Pool
from sewar.full_ref import uqi
from sewar.no_ref import d_lambda ,qnr, d_s
import numpy as np
import cv2
from imutils import paths
import argparse
from PIL import Image
import pytesseract
from skimage.filters import threshold_otsu
import cpbd
from scipy import ndimage
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from skimage.restoration import estimate_sigma
import hurry
from hurry.filesize import size , alternative

output_txt = 'output_txt/'
#binary_image = '/Users/harishkhollam/Documents/VirtualEnvs/Testing-sewar/binary_image/'
dpi = 80

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

# Divide the argument list into single image path
def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

# Donoho's method based on wavelet
# http://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.estimate_sigma
# It also works with color images, you just need to set multichannel=True and average_sigmas=True
# estimate_sigma(img, multichannel=True, average_sigmas=True)

# print('Estimating Value of sigma : High numbers mean low noise.')
# print('cumulative probability of blur detection : lower the score more blur the image and less sharper')
# print('Low number means lot of blur')

def multiprocessing_func(imagePath):
    # load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
  	# # method
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	# text = "Not Blurry"
	# print('Sharpness : ',cpbd.compute(gray))
	# print('Sigma for noise :',estimate_sigma(image, multichannel=True, average_sigmas=True))
 
	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	# if fm < args["threshold"]:
	# 	text = "Blurry"
	# print('image is {} Blur Score : {:.2f}\n'.format(text,fm))

	# Get Height & Width of image
	height, width, depth = image.shape
	# figsize = width / float(dpi), height / float(dpi)
	raster_size = os.path.getsize(imagePath)
	raster_size = size(raster_size, system=alternative)
	metrics = raster_size.split(' ')
	if metrics[1] == 'KB':
		metrics[0] =  int(metrics[0])/1000
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#thresh = threshold_otsu(gray)
	# binary = image >= thresh
	gray_text = str(pytesseract.image_to_string(gray))
	filename = imagePath.split('/')
	name = filename[1].split('.')
	f = open(output_txt+name[0]+'_gray.txt', 'w+' )
	f.write(gray_text)
	f.flush()
	f.seek(0)
	first = f.read(1)
	if not first:
		Tesseract_success = '0'
	else:
		Tesseract_success = '1'
	f.close()
	# print('{},{},{},{},{:.6f},{:.6f},{:.2f},{}'.format(imagePath,raster_size,width,height,cpbd.compute(gray),estimate_sigma(image, multichannel=True, average_sigmas=True),fm,0))
	print('{},{},{},{},{:.6f},{}'.format(filename[1],metrics[0],width,height,estimate_sigma(image, multichannel=True, average_sigmas=True),Tesseract_success))

	
	# 	binary_image_path = str(binary_image)+str(name[0])+".png"
	# 	
	# # 	binary[binary == True] == True
	# # 	binary = binary.astype(np.uint8)
	# 	
	# 	print(binary)

		#
	# 	binary[binary == True] == 0
	# 	binary[binary == False] == 255
	# 	binary = binary.astype(np.uint8)
		#
	# 
	# 	# e.g. when writing using OpenCV
	# 	
	# 	cv2.imwrite(binary_image_path, binary)
		# 	
	# 	plt.figure(figsize = figsize)
	# 	plt.imsave(binary_image_path, binary, cmap=cm.gray)
		#


		#
	# 	binary_text = str(pytesseract.image_to_string(binary_image_path))
	# 	f = open(output_txt+name[0]+'_binary.txt', 'w' )
	# 	f.write(binary_text)
	# 	f.close()
		#
	
	
	
		# show the image
		#cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
		#cv2.imshow("Image", image)
		#key = cv2.waitKey(0)

if __name__ == '__main__':
	starttime = time.time()
	# Add Headers only to print
	#print('Image,Width,Hight,Sharpness,Noise(Sigma),Blur Score,Tesseract_Success')
	print('Image,Size,Width,Hight,Noise(Sigma),Tesseract_Success')
	processes = []
	x = list(divide_chunks(list(paths.list_images(args["images"])), 10))
	for list in x:
		for imagePath in list:
			p = multiprocessing.Process(target=multiprocessing_func, args=(imagePath,))
			processes.append(p)
			p.start()
		for process in processes:
			process.join()
	#Print following command to check the time required to execute the code
	# print('That took {} seconds'.format(time.time() - starttime))
