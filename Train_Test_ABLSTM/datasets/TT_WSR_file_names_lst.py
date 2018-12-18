# -*- coding: utf-8 -*-
import os
import glob
import shutil  
import random
import os.path

# test_GV2_HIKMPPU.lst
# train_GV2_HIKMPPU.lst
rootdir = r"./Total_Text_WSR/"       # 指明被遍历的文件夹
file_name = open(r"./Total_Text_WSR/Total_Text_WSR.lst", 'w')
for parent, dirnames, filenames in os.walk(rootdir):   #1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字

	for dirname in  dirnames:        #输出文件夹信息
		ori_lists = glob.glob(parent + '/'+dirname+ '/*.jpg')		
		mask_lists = glob.glob(parent + '/'+ dirname+'_gt'+'/*.png')
		random.shuffle(ori_lists)
		length = len(rootdir)
		length_p =len(rootdir)		
		for i in range(len(ori_lists)):
			lst = (ori_lists[i][length:]).split('/')
			# print len(lst), lst	
			# lst[0] : one space
			img = lst[0] + "/" + lst[1] +"/"+ lst[2]+ "/" + lst[3]		 
			img_gt = lst[0] + "/" + lst[1]+"/" + lst[2] + "_gt" + "/" + lst[3][:-4] +'.png'				 
			one_record = img + " " + img_gt 

			# one_record = img 
			file_name.write(one_record+"\n")
file_name.close()
print "Train pairs list, All Done"


# rootdir = r"I:\Dataset_My_make\KAIST_Merage\Rotate\scale_keep_size_700\test0"       # 指明被遍历的文件夹
# file_name = open(r"I:\Dataset_My_make\KAIST_Merage\Rotate\scale_keep_size_700\test.lst", 'w')

# for parent, dirnames, filenames in os.walk(rootdir):   #1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字

# 	for dirname in  dirnames:        #输出文件夹信息
# 		ori_lists = glob.glob(parent + '\\'+dirname+ '\*.jpg')		
# 		mask_lists = glob.glob(parent + '\\'+ dirname+'_gt'+'\*.png')
# 		random.shuffle(ori_lists)
# 		length = len(rootdir)
# 		length_p =len(rootdir)		
# 		for i in range(len(ori_lists)):
# 			lst = (ori_lists[i][length:]).split('\\')	
# 			img = lst[1] + "/" + lst[2] +"/"+ lst[3]+ "/" + lst[4]	
# 			img_gt = lst[1] + "/" + lst[2]+"/" + lst[3] + "_gt" + "/" + lst[4][:-4] +'.png'				 
# 			# one_record = img + " " + img_gt 
# 			one_record = img 
# 			# one_record = img 
# 			file_name.write(one_record+"\n")
# file_name.close()
# print "Test list, All Done"
