import numpy as np
import copy
import cv2
import os
from math import pow, sqrt
# # Find the intersection of two arrays
def multidim_intersect(arr1, arr2):
   arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
   arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
   intersected = np.intersect1d(arr1_view, arr2_view)
   return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

def box_distance(rect_rr):
   box_rr  = cv2.cv.BoxPoints(rect_rr )
   box_rr  = np.int0(box_rr )
   c_box_rr  = np.zeros(box_rr .shape, dtype=np.int32)
   c_box_rr [:, 1] = box_rr [:, 0]
   c_box_rr [:, 0] = box_rr [:, 1]
   lst_rr = c_box_rr.tolist()
   lst_rr_st = sorted(lst_rr)
   x0 = lst_rr_st[0][0]; y0 = lst_rr_st[0][1]
   x1 = lst_rr_st[1][0]; y1 = lst_rr_st[1][1]
   x2 = lst_rr_st[2][0]; y2 = lst_rr_st[2][1]

   d_x1 = (x0 - x1)
   d_y1 = (y0 - y1)
   dis1 = sqrt(pow(d_x1, 2) + pow(d_y1, 2))

   d_x2 = (x0 - x2)
   d_y2 = (y0 - y2)
   dis2 = sqrt(pow(d_x2, 2) + pow(d_y2, 2))

   return int(dis1), dis2
def Union_wib_bs_region_b(wib_block_yxs, region_block_yxs):
   union_wib_bs_region_b =[]
   for wib_b in wib_block_yxs:
      wib_b_yx = wib_b.copy()
      for region_b in region_block_yxs:
         region_b_yx = region_b.copy()
         ret_wb_rb = multidim_intersect(wib_b_yx, region_b_yx)
         # print "ret_wb_rb is: ", len(ret_wb_rb)

         if len(ret_wb_rb)!=0:
            wb = wib_b_yx.tolist()
            rb = region_b_yx.tolist()
            temp = wb + rb
      temp =np.array(temp)
      union_wib_bs_region_b.append(temp)
   return union_wib_bs_region_b

def union_grop_words_block(union_wb_rb_img, wib_block_yxs, remain_in_blocks, region_block_yxs, img_name, save_path):
# If the word and text block and returns the intersection
   rr_img = copy.copy(union_wb_rb_img)
   h_img = copy.copy(union_wb_rb_img)
   h, w, _ =rr_img.shape

   det_save_path =  save_path + '\\'+'det_result'
   if not os.path.exists(det_save_path):
      os.makedirs(det_save_path)

   save_img = det_save_path+'\\' + img_name[:-4]+'_det.jpg'
   det_file = det_save_path+'\\' + img_name[:-4]+'_det.gt'
   det_file01 = det_save_path+'\\' + img_name[:-4]+'.lst'
   save_file = open(det_file, 'w')
   save_file01 = open(det_file01, 'w')   

   # save_h_img = det_save_path+'\\' + 'res_'+img_name[:-4]+'.tif'  
   # h_det_file = det_save_path+'\\' +'res_' + img_name[:-4]+'.txt'   
   # save_h_file= open(h_det_file, 'w')

   save_m_img = det_save_path+'\\' + 'res_'+img_name[:-4]+'.tif'  
   m_det_file = det_save_path+'\\' +'res_' + img_name[:-4]+'.txt'   
   save_m_file= open(m_det_file, 'w')  

   union_wib_bs_region_b =  Union_wib_bs_region_b(wib_block_yxs, region_block_yxs)

   for i in range(len(union_wib_bs_region_b)):

      rr_region = remain_in_blocks[i]      
      rect_rr = cv2.minAreaRect(rr_region)
      (x_rr , y_rr ),(width_rr , height_rr ), theta_rr  = rect_rr 
      dis_rr1, dis_rr2 = box_distance(rect_rr)

      union_region = union_wib_bs_region_b[i]
      rect = cv2.minAreaRect(union_region)
 
      (x, y),(width, height), theta = rect
      dis1, dis2 = box_distance(rect)


      box = cv2.cv.BoxPoints(rect)
      box = np.int0(box)
      c_box = np.zeros(box.shape, dtype=np.int32)
      c_box[:, 1] = box[:, 0]
      c_box[:, 0] = box[:, 1]
      cv2.drawContours(rr_img, [c_box], 0, (255, 0, 0), 10)

      r_h = height/ height_rr 
      r_w = width/ width_rr

      if dis_rr1 > dis_rr2:
         if abs(dis_rr1 - width_rr)<=1:            
            if r_h>=2:
               print "***01"
               rect_rr = ((x_rr, y_rr),(width_rr, height_rr* 4/3), theta_rr)
            else:
               print "***02"
               rect_rr = ((x, y),(width, height), theta)
         elif abs(dis_rr1 - height_rr)<=1:            
            if r_w>=2:
               print "***03"
               rect_rr = ((x_rr, y_rr),(width_rr* 4/3, height_rr), theta_rr)
            else:
               print "***04"
               rect_rr = ((x, y),(width, height), theta)
         else:
            if r_w>=2:
               print "***05"
               # rect_rr = ((x, y),(width, height), theta)
               rect_rr = ((x_rr, y_rr),(width_rr* 4/3, height_rr), theta_rr)
            else:
               print "***06"
               # rect_rr = ((x, y),(width, height), theta)
               rect_rr = ((x_rr, y_rr),(width_rr, height_rr* 4/3), theta_rr)


      elif dis_rr1 <= dis_rr2:
         if abs(dis_rr1 - width_rr)<=1:            
            if r_w>=2:
               print "***07" 
               rect_rr = ((x_rr, y_rr),(width_rr* 5/3, height_rr), theta_rr)
            else:
               print "***8"
               rect_rr = ((x, y),(width, height), theta)                              
         elif abs(dis_rr1 - height_rr)<=1:
            
            if r_h>=2:
               print "***9" 
               rect_rr = ((x_rr, y_rr),(width_rr, height_rr* 5/3), theta_rr)
            else:
               print "***10" 
               rect_rr = ((x, y),(width, height), theta)
         else:            
            if r_h>=2:
               print "***11" 
               rect_rr = ((x_rr, y_rr),(width_rr, height_rr* 4/3), theta_rr)
            else:
               print "***12" 
               # rect_rr = ((x_rr, y_rr),(width_rr* 4/3, height_rr), theta_rr)
               rect_rr = ((x, y),(width, height), theta)

      box_rr  = cv2.cv.BoxPoints(rect_rr )
      box_rr  = np.int0(box_rr )
      c_box_rr  = np.zeros(box_rr .shape, dtype=np.int32)
      c_box_rr [:, 1] = box_rr [:, 0]
      c_box_rr [:, 0] = box_rr [:, 1]
      cv2.drawContours(union_wb_rb_img, [c_box_rr ], 0, (0,255, 255), 5)
      ctr_x = rect_rr[0][0]
      ctr_y = rect_rr[0][1]
      one_w =  rect_rr[1][0]
      one_h = rect_rr[1][1]
      one_theta =  rect_rr[2]
      one_nms =  str(int(ctr_x))+","+str(int(ctr_y))+","+str(int(one_w))+","+str(int(one_h))+","+str(round(one_theta, 2))
      save_file01.write(one_nms+"\n")

      # cv2.drawContours(union_wb_rb_img, [c_box_rr ], 0, ((255,0,0)), 3) 
           
      # win1 = cv2.namedWindow("I", flags =0)
      # cv2.imshow("I", union_wb_rb_img)
      # cv2.waitKey(0)

      one_record = str(int(x))+","+str(int(y))+","+str(int(width))+","+str(int(height))+","+str(round(theta, 2))
      save_file.write(one_record+"\n")


      x1 = c_box_rr[1][0]; y1= c_box_rr[1][1]
      x2= c_box_rr[2][0]; y2 = c_box_rr[2][1]
      x3 = c_box_rr[3][0]; y3 = c_box_rr[3][1]
      x4 = c_box_rr[0][0]; y4 = c_box_rr[0][1]

      one_m_record = str(int(x1))+","+str(int(y1))+","+str(int(x2))+","+str(int(y2))+","+str(int(x3))+","+str(int(y3))+","+str(int(x4))+","+str(int(y4))
      save_m_file.write(one_m_record+"\n")
      # print "****************one_m_record is ****************", one_m_record

   # win1 = cv2.namedWindow("I", flags =0)
   # cv2.imshow("I", union_wb_rb_img)
   # cv2.waitKey(0)

   # union_wb_rb_img = cv2.resize(union_wb_rb_img,(int(0.5*union_wb_rb_img.shape[1]),int(0.5*union_wb_rb_img.shape[0])))   
   cv2.imwrite(save_img, union_wb_rb_img)
   # cv2.imwrite(save_h_img, h_img)
   # save_h_file.close()
   save_file.close()
   save_file01.close()
   save_m_file.close()

