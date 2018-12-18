import numpy as np
import copy
import cv2
import os
from Union_group_wrods_block import union_grop_words_block

def merage_words(matchs, region_word_yxs, region_block_yxs):
# The pixels in the word are added to the corresponding text block area
# The high is bigger than the word and block

	for match in matchs:
		i = match[0]
		j = match[1]
		region_block_yxs[j] = np.append(region_word_yxs[i], region_block_yxs[j], axis=0)
		#  The union of word and block		
	return region_block_yxs

def words_in_block_idx(matchs, region_block_yxs):

	# generate empty list to store the meraged word region	
	words_in_blocks_idx = []
	for m in range(len(region_block_yxs)):
		wib =[]
		words_in_blocks_idx.append(wib)

	for i in range(len(matchs)):
		m = matchs[i][0]
		n = matchs[i][1]
		words_in_blocks_idx[n].append(matchs[i])

		for j in range(len(matchs)):
			if  n == matchs[j][1]:
				# print n, j
				words_in_blocks_idx[n].append(matchs[j])
		# print "M[n] is: ", M[n]
		temp = []
		for i in words_in_blocks_idx[n]:
			if i not in temp:
				temp.append(i)
		if temp !=[]:
			words_in_blocks_idx[n] = temp
		else: 
			pass
	# print "*****************************", words_in_blocks

# Delte the element of []
	temp_wib =[]		
	for wib in words_in_blocks_idx:
		if wib !=[]:
			temp_wib.append(wib)
	words_in_blocks_idx = temp_wib
	return words_in_blocks_idx

def word_in_block_word_union_block(img, matchs, region_word_yxs, region_block_yxs, img_name, save_path):
	union_wb_rb_img = copy.copy(img)
	words_in_blocks_idx = words_in_block_idx(matchs, region_block_yxs)
	ws_in_b, remain_in_blocks = group_words_in_block(img, words_in_blocks_idx, region_word_yxs, region_block_yxs, img_name, save_path)
	union_grop_words_block(union_wb_rb_img, ws_in_b, remain_in_blocks, region_block_yxs, img_name, save_path)
	# win = cv2.namedWindow("w_in_b_rect", flags = 0)
	# cv2.imshow("w_in_b_rect", w_in_b_rect)
	# cv2.waitKey(0)

def group_words_in_block(img, words_in_blocks_idx, region_word_yxs, region_block_yxs, img_name, save_path):
	# Store the words or chars which in the same text block
	ws_in_b=[]
	wib_img =copy.copy(img)
	h_img = copy.copy(img)
	# print "words_in_blocks_idx is: ", words_in_blocks_idx
	det_save_path =  save_path + '/'+'words_in_block'
	if not os.path.exists(det_save_path):
		os.makedirs(det_save_path)

	save_img = det_save_path+'/' + img_name[:-4]+'_word.jpg'
	det_file = det_save_path+'/' + img_name[:-4]+'_word.gt'
	save_h_img = det_save_path+'/' + 'res_'+img_name[:-4]+'.tif'	
	h_det_file = det_save_path+'/' +'res_' + img_name[:-4]+'.txt'

	save_file = open(det_file, 'w')
	save_h_file= open(h_det_file, 'w')

	block_idx_temp = []
	for i in range(len(words_in_blocks_idx)):
		temp =[]
		for j in range(len(words_in_blocks_idx[i])):
			word_idx = words_in_blocks_idx[i][j][0]
			block_idx = words_in_blocks_idx[i][j][1]
			block_idx_temp.append(block_idx)
			temp = temp + region_word_yxs[word_idx].tolist()
		ws_in_b.append(temp)
	block_idx_temp = list(set(block_idx_temp))

	remain_in_blocks = []
	for i in block_idx_temp:
		remain_in_blocks.append(region_block_yxs[i])


	ws_in_b_change =[]
	for lst in ws_in_b:
		arr = np.array(lst)
		ws_in_b_change.append(arr)
	ws_in_b= np.array(ws_in_b_change)



	for word in ws_in_b:
		rect = cv2.minAreaRect(word)
		(x, y),(width, height), theta = rect
		box = cv2.cv.BoxPoints(rect)
		box = np.int0(box)

		c_box = np.zeros(box.shape, dtype=np.uint64)
		c_box[:, 1] = box[:, 0]
		c_box[:, 0] = box[:, 1]
		cv2.drawContours(wib_img, [c_box], 0, (255, 0, 0), 5)
		# one_record = str(int(x))+","+str(int(y))+","+str(int(width))+","+str(int(height))+","+str(round(theta, 2))
		# save_file.write(one_record+"\n")
		x1 = c_box[1][0]; y1= c_box[1][1]
		x2= c_box[2][0]; y2 = c_box[2][1]
		x3 = c_box[3][0]; y3 = c_box[3][1]
		x4 = c_box[0][0]; y4 = c_box[0][1]

		one_record = str(int(x1))+","+str(int(y1))+","+str(int(x2))+","+str(int(y2))+str(int(x3))+","+str(int(y3))+","+str(int(x4))+","+str(int(y4))
		save_file.write(one_record+"\n")


		Ys = [i[0] for i in box]
		Xs = [i[1] for i in box]
		x1 = min(Xs); x2 = max(Xs)
		y1 = min(Ys); y2 = max(Ys)
		cv2.rectangle(h_img, (x1,y1),(x2, y2),(0,255,255),5)
		h_one_record = str(int(x1))+","+str(int(y1))+","+str(int(x2))+","+str(int(y2))
		save_h_file.write(h_one_record+"\n")

	# wib_img = cv2.resize(wib_img,(int(0.5*wib_img.shape[1]),int(0.5*wib_img.shape[0])))
	# h_img = cv2.resize(h_img,(int(0.5*h_img.shape[1]),int(0.5*h_img.shape[0])))
	cv2.imwrite(save_img, wib_img)
	cv2.imwrite(save_h_img, h_img)		

	# save_file = open(det_file, 'w')
	# save_h_file= open(h_det_file, 'w')
	save_file.close()
	save_h_file.close()	
	return ws_in_b, remain_in_blocks
