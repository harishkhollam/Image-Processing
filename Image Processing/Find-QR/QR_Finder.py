import numpy as np
import cv2
import time
import os
import multiprocessing 
from multiprocessing import Pool
from imutils import paths
import argparse
from pyzbar import pyzbar


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
args = vars(ap.parse_args())

def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def display(im, bbox):
    n = len(bbox)
    for j in range(n):
        cv2.line(im, tuple(bbox[j][0]), tuple(bbox[ (j+1) % n][0]), (255,0,0), 3)

    # Display results
    cv2.imshow("Results", im)

def get_info(image):
	inputImage = cv2.imread(image)
	qrDecoder = cv2.QRCodeDetector()
	# Detect and decode the qrcode
	barcodes = pyzbar.decode(inputImage)
# loop over the detected barcodes
	for barcode in barcodes:
		# extract the bounding box location of the barcode and draw the
		# bounding box surrounding the barcode on the image
		(x, y, w, h) = barcode.rect
		cv2.rectangle(inputImage, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
		# the barcode data is a bytes object so if we want to draw it on
		# our output image we need to convert it to a string first
		barcodeData = barcode.data.decode("utf-8")
		barcodeType = barcode.type
 
		# draw the barcode data and barcode type on the image
		text = "{} ({})".format(barcodeData, barcodeType)
		cv2.putText(inputImage, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 0, 255), 2)
 
		# print the barcode type and data to the terminal
		image_name = image.split('/')
		print("{},{},{}".format(image_name[1],barcodeType, barcodeData))
		
  		# show the output image
		#cv2.imshow("Image", inputImage)
		#cv2.waitKey(0)
 
	


if __name__ == '__main__':
	starttime = time.time()
	print('Image,Barcode_Type,Barcode_Data')
	processes = []
	x = list(divide_chunks(list(paths.list_images(args["images"])), 10)) 
	for list in x:
		for imagePath in list:
			p = multiprocessing.Process(target=get_info, args=(imagePath,))
			processes.append(p)
			p.start()
		for process in processes:
			process.join()
 	# print('That took {} seconds'.format(time.time() - starttime))