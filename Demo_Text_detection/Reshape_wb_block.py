# -*- coding: utf-8 -*-
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import copy
import os
from os.path import join
import cv2

def reshape_I(img, word, block):

	if word.shape[0] != img.shape[0] or word.shape[1] != img.shape[1]:
		word = cv2.resize(word, (img.shape[1], img.shape[0]))
	else:
		pass

	if block.shape[0] != img.shape[0] or block.shape[1] != img.shape[1]:
		block = cv2.resize(block, (img.shape[1], img.shape[0]))
	else:
		pass
	# word[:, 0:10]= 0
	# word[0:10, :]= 0
	# word[:, (word.shape[1]-10):word.shape[1]]= 0
	# word[(word.shape[0]-10):word.shape[0], :]= 0
		
	# block[:, 0:10]= 0
	# block[0:10, :]= 0
	# block[:, (block.shape[1]-10):block.shape[1]]= 0
	# block[(block.shape[0]-10):block.shape[0], :]= 0
	
	return word, block

def label_word_block(word, block):
	# 
	
	th_w = threshold_otsu(word)
	bw_w = closing(word > th_w, square(1))

	th_b = threshold_otsu(block)
	bw_b = closing(block > th_b, square(1))

	# label image regions
	label_word = label(bw_w)
	label_block = label(bw_b)

	image_label_overlay_w = label2rgb(label_word, image=word)
	image_label_overlay_b = label2rgb(label_block, image=block)

	return label_word, label_block