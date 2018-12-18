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
from Reshape_wb_block import reshape_I, label_word_block
from Plot_rects import plot_rects
from Word_in_block_word_union_block import  merage_words, word_in_block_word_union_block

def rect_block(img, label_block, img_name, save_path):
	# Output the block rect
	# print "label_block img_name is", img_name
	save_w_b_path = save_path+'/words_blocks'
	if not os.path.exists(save_w_b_path):
		os.makedirs(save_w_b_path)	

	block_img = copy.copy(img)
	region_block_yxs = []
	for region in regionprops(label_block):

		if region.area >= 200:
			# print region.area
			coords = region.coords
			region_block_yxs.append(coords)

			rect = cv2.minAreaRect(coords)
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)
			c_box = np.zeros(box.shape, dtype=np.uint64)
			c_box[:, 1] = box[:, 0]
			c_box[:, 0] = box[:, 1]
			cv2.drawContours(block_img, [c_box], 0, (255, 0, 0), 5)

		# win1 = cv2.namedWindow('rect_block', flags=0)
		# cv2.imshow("rect_block", block_img)
		# cv2.waitKey(0) 
	block_save = save_w_b_path + '/' + img_name[:-4] + '_b.tif'
	# block_img = cv2.resize(block_img,(int(0.5*block_img.shape[1]),int(0.5*block_img.shape[0])))   
	cv2.imwrite(block_save, block_img)
	return region_block_yxs

def rect_word(img, label_word, img_name, save_path):
	# print "label_word img_name is", img_name
	#Output the word rect
	save_w_b_path = save_path+'/words_blocks'
	if not os.path.exists(save_w_b_path):
		os.makedirs(save_w_b_path)	
		
	word_img = copy.copy(img)
	ct_word_yxs = []
	region_word_yxs = []
	region_word_boxes = []
	for region in regionprops(label_word):
		# take regions with large enough areas
		if region.area >= 10:
			coords = region.coords
			region_word_yxs.append(coords)

			rect = cv2.minAreaRect(coords)
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)
			c_box = np.zeros(box.shape, dtype=np.uint64)
			c_box[:, 1] = box[:, 0]
			c_box[:, 0] = box[:, 1]
			cv2.drawContours(word_img, [c_box], 0, (0,255,255), 5)

	# win1 = cv2.namedWindow('rect_word', flags=0)
	# cv2.imshow("rect_word", word_img)
	# cv2.waitKey(0)
	# cv2.imwrite(r"C:\Users\Administrator\Desktop\KAIT_EKM01\a.jpg", word_img)

	word_save = save_w_b_path + '/' + img_name[:-4] + '_w.tif'
	# word_img = cv2.resize(word_img,(int(0.5*word_img.shape[1]),int(0.5*word_img.shape[0])))   
	cv2.imwrite(word_save, word_img)
	return  region_word_yxs


# Find the intersection of two arrays
def multidim_intersect(arr1, arr2):
	arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
	arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
	intersected = np.intersect1d(arr1_view, arr2_view)
	return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

# The index of a word and its index in the region of the text
def word_index_block_index(region_word_yxs, region_block_yxs):

	i = 0  # word
	matchs = []
	for word_yxs in region_word_yxs:
		j = 0  # block
		for block_yxs in region_block_yxs:
			word_yxs = word_yxs.copy()
			block_yxs = block_yxs.copy()
			# if  [val for val in a if val in b]:
			inters = multidim_intersect(word_yxs, block_yxs)
			
			if len(inters):
				match = [i, j]
				matchs.append(match)
			else:
				pass
			j = j + 1
		i = i + 1
	return matchs
