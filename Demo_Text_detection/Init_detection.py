import numpy as np
import copy
import cv2
import os
from Reshape_wb_block import reshape_I, label_word_block
from Plot_rects import plot_rects
from Word_in_block_word_union_block import  merage_words, word_in_block_word_union_block
from Word_block_rect import rect_block, rect_word, word_index_block_index


def word_block_analysis(img, img_name, save_path, word, block):	
		w = copy.copy(word)
		b = copy.copy(block)
		save_img_path = save_path + '\det_result'
		if not os.path.exists(save_img_path):
			os.makedirs(save_img_path)
		else:
			pass
		print "**************"

		det_save_path =  save_path + '/'+'det_result'
		if not os.path.exists(det_save_path):
			os.makedirs(det_save_path)

		save_img = det_save_path+'/' + img_name[:-4]+'_det.jpg'
		det_file = det_save_path+'/' + img_name[:-4]+'_det.gt'
		save_h_img = det_save_path+'/' + 'res_'+img_name[:-4]+'.tif' 
		h_det_file = det_save_path+'/' +'res_' + img_name[:-4]+'.txt'
		save_file = open(det_file, 'w')
		save_h_file= open(h_det_file, 'w')



		if (255 in w[:, :]) and (255 in b[:, :]):
			label_word, label_block = label_word_block(word, block)
			region_block_yxs = rect_block(img, label_block, img_name, save_path)
			region_word_yxs = rect_word(img, label_word, img_name, save_path)
			
			# matchs = word_index_block_index(ct_word_yxs, region_block_boxes)
			matchs = word_index_block_index(region_word_yxs, region_block_yxs)
			# print "matchs is:", matchs

			# The word which is inclued in the block
		
			word_in_block_word_union_block(img, matchs, region_word_yxs, region_block_yxs, img_name, save_path)		
 
			# region_block_yxs_ori = copy.copy(region_block_yxs)			
			# region_block_yxs_mer = merage_words(matchs, region_word_yxs, region_block_yxs)
			# det_img = plot_rects(img, region_block_yxs_ori, region_block_yxs_mer, txt_file)
			# cv2.imwrite(save_img, det_img)
		else:
			img = cv2.resize(img,(int(0.5*img.shape[1]),int(0.5*img.shape[0])))
			cv2.imwrite(save_img, img)	
			save_file.close()
			save_h_file.close()
		# txt_file.close()