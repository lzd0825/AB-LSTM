# -*- coding: utf-8 -*-
import numpy as np
import cv2, os
import math
from math import cos, sin,tan, sqrt

def endWith(s, *endstring):
  array = map(s.endswith, endstring)
  if True in array:
    return True
  else:
    return False 
    
# 根据Total_text数据集给定的X,Y列表，生成缩放的多边形, 由多边形生成区域
def draw_counter(lst, Half_map, h, w):
    # print "This is ", lst
    max_num=0
    for i in range(0, len(lst)):
      one_lst = lst[i].strip('\n')
      all_xy = one_lst.split(',')
      if  len(all_xy)>max_num:
        max_num=len(all_xy)

      lg=len(one_lst.split(','))
      # print lg       
      for j in range(0,len(all_xy)/2-1, 2):
        x1=int(all_xy[j])
        y1=int(all_xy[j+1])
        x_n1=int(all_xy[j+2])
        y_n1=int(all_xy[j+3])

        y2= int(all_xy[lg-j-1])
        x2=int(all_xy[lg-j-2])

        y_n2= int(all_xy[lg-j-3])
        x_n2=int(all_xy[lg-j-4])
        x_e1 = int(all_xy[0])
        y_e1 = int(all_xy[1])
        x_e2 = int(all_xy[lg-2])
        y_e2 =int(all_xy[lg-1])

        lambda1=1*1.0/3; lambda2=3
        x01 = int((x1+lambda2*x2)/(1+lambda2)); y01 = int((y1+lambda2*y2)/(1+ lambda2))
        x02 = int((x1+lambda1*x2)/(1+lambda1)); y02 = int((y1+lambda1*y2)/(1+lambda1))
        # cv2.circle(Half_map, (int(x01), int(y01)), 5, ( 255,0, 0), -1)
        # cv2.circle(Half_map, (int(x02), int(y02)), 5, ( 255,0, 0), -1)


        x_0n1=int(all_xy[j+2])
        y_0n1=int(all_xy[j+3])
        y_0n2= int(all_xy[lg-j-3])
        x_0n2=int(all_xy[lg-j-4])

        x_0n1 = int((x_n1+lambda2*x_n2)/(1+lambda2)); y_0n1 = int((y_n1+lambda2*y_n2)/(1+ lambda2))
        x_0n2 = int((x_n1+lambda1*x_n2)/(1+lambda1)); y_0n2 = int((y_n1+lambda1*y_n2)/(1+lambda1))
        x_0e1 = int((x_e1+lambda2*x_e2)/(1+lambda2)); y_0e1 = int((y_e1+lambda2*y_e2)/(1+ lambda2))
        x_0e2 = int((x_e1+lambda1*x_e2)/(1+lambda1)); y_0e2 = int((y_e1+lambda1*y_e2)/(1+lambda1))
  
        cv2.line(Half_map,(x01, y01),(x_0n1, y_0n1),(255,255,255),2)
        cv2.line(Half_map,(x02, y02),(x_0n2, y_0n2),(255,255,255),2)
        cv2.line(Half_map,(x_0e1, y_0e1),(x_0e2, y_0e2 ),(255,255,255),2)
    # cv2.imshow("Half_map", Half_map) 
    # cv2.waitKey(0)
    print "The num of points is:", max_num
    return max_num

def xy_strings(img, gtfile):
  gtlines = gtfile.readlines()
  gtlines_len = len(gtlines)  
  xy_strs=[]
  all_xs=[]
  all_ys=[]
  all_xy = []
  for i in range(gtlines_len):
    x1, y1, x4, y4, x3, y3, x2, y2= gtlines[i].split(",")
    all_xs.append([x1, x2, x3, x4])
    all_ys.append([y1, y2, y3, y4])   

  for j in range(len(all_xs)): 
    one=''
    for k in range(len(all_xs[j])):
      x=all_xs[j][k]
      y=all_ys[j][k]

      cv2.circle(img,(int(x), int(y)),5,(0,255,255),-1) 

      if one=='':
        one=x+','+y
      else:
        one=one+','+x+','+y
    xy_strs.append(one)
  return xy_strs

def generate_half_polygons(gts_path, img_path, save_half_polygons_path):
  if not os.path.exists(save_half_polygons_path):
    os.makedirs(save_half_polygons_path)

  max_nmus =[]
  for gt in os.listdir(gts_path):
    if endWith(gt, '.txt'):
      gtfile = open(gts_path + "/" + gt, 'r')
      save_map=save_half_polygons_path + r"/" +gt[:-4]+".tif"
      save_img=save_half_polygons_path + r"/" +gt[:-4]+".jpg"
      img_file = img_path + "/" + gt[:-4]+".jpg"  
      print img_file  
      img = cv2.imread(img_file)
      h, w, _ = img.shape
      Half_map = np.zeros((img.shape[0], img.shape[1]))
      gtlines =xy_strings (img, gtfile)

      maxnum=draw_counter(gtlines, img, h, w)
      maxnum=draw_counter(gtlines, Half_map, h, w)
      cv2.imwrite(save_img, img)
      cv2.imwrite(save_map, Half_map)
      max_nmus.append(maxnum)
      gtfile.close()

  print "All Done"
  print "--------------------------"
  print "max_nmus is: ", max_nmus
  print "max_nmus is :", max(max_nmus)


def generate_half_regions(ori_masks, save_half_regions):

  if not os.path.exists(save_half_regions):
    os.makedirs(save_half_regions) 

  masks = [flg for flg in os.listdir(ori_masks) if '.tif' in flg]
  for msk in masks: 
    msk_file = ori_masks+'/'+msk
    save_file = save_half_regions+'/'+msk[:-4]+'.png'
    img=cv2.imread(msk_file)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,80, 255, cv2.THRESH_BINARY)
    # 识别轮廓
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img = np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8)
    for cnt in contours:
      
      # 近似多边形，True表示闭合
      approx = cv2.approxPolyDP(cnt, 1, True)   
      hull = cv2.convexHull(cnt)
      cv2.drawContours(img, [approx], -1, (255, 255, 255), -1) 
    cv2.imwrite(save_file, img)
    print "Next one"
  print "All Done"


def MSRA_TD500(gt_path,img_path,save_four_gt):
    if not os.path.exists(save_four_gt):
      os.makedirs(save_four_gt)

    if not os.path.exists(save_images):
        os.makedirs(save_images)
        
    for gt in os.listdir(gt_path):
        if endWith(gt, '.gt'):
            gtfile = open(gt_path + "/" + gt, 'r')
            four_point_file = open(save_four_gt + "/" + gt[:-3]+'.txt', 'w')
            img_file=img_path+"/"+gt[:-3]+'.JPG'
            print ">>>",img_file
            img = cv2.imread(img_file)

            save_img=save_images+"/"+gt[:-3]+'.jpg'

            cv2.imwrite(save_img, img)

            gtlines = gtfile.readlines()
            gtlines_len = len(gtlines)
            dxys=[]
            for i in range(gtlines_len):
                dxy0=[]
                c1,c2,ctr_x,ctr_y,w,h,theta= gtlines[i].split(" ")
                # ctr_x,ctr_y,w,h,theta= gtlines[i].split(" ")
                ctr_x=int(ctr_x);ctr_y=int(ctr_y);
                w=float(w);h=float(h);
                theta=float(theta)
                # print theta
                if theta<0:
                    k=tan(theta)   
                    theta=abs(theta)                             
                    x0=(ctr_x+w*0.5)-0.5*(w*cos(theta)+h*sin(theta))                
                    y0=(ctr_y+h*0.5)+0.5*(w*sin(theta)-h*cos(theta)) 
                    x1=x0+h*sin(theta) 
                    y1=y0+h*cos(theta)
                    x2=x0+h*sin(theta)+w*cos(theta) 
                    y2=y0+h*cos(theta)-w*sin(theta)
                    x3=x0+w*cos(theta)
                    y3=y0-w*sin(theta)

                else:
                    k=tan(theta)
                    x0=(ctr_x+w*0.5)-0.5*(w*cos(theta)-h*sin(theta));
                    y0=(ctr_y+h*0.5)-0.5*(w*sin(theta)+h*cos(theta));
                    x1=x0-h*sin(theta);
                    y1=y0+h*cos(theta);
                    x2=x0-h*sin(theta)+w*cos(theta);
                    y2=y0+h*cos(theta)+w*sin(theta);
                    x3=x0+w*cos(theta) 
                    y3=y0+w*sin(theta);

                x0=int(x0); y0=int(y0)
                x1=int(x1); y1=int(y1)
                x2=int(x2); y2=int(y2)
                x3=int(x3); y3=int(y3)

                cv2.line(img,(x0,y0),(x1,y1),(0,255,255),3)            
                cv2.line(img,(x1,y1),(x2,y2),(0,255,255),3) 
                cv2.line(img,(x2,y2),(x3,y3),(0,255,255),3) 
                cv2.line(img,(x3,y3),(x0,y0),(0,255,255),3) 

                one_record = str(x0)+","+str(y0)+","+ str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+str(x3)+","+str(y3)
                four_point_file.write(one_record+"\n")
            # cv2.imwrite(save_img, img)
            gtfile.close()
            four_point_file.close()
        else:
            continue
        print "Next"
    print "over all"

if __name__ == "__main__":

  gt_path=r"./train_gt_ori"
  save_four_gt = r'./Train_gt'
  img_path=r"./train_ori"
  save_images=r"./Train"

  gts_path = r'./Train_gt'
  save_half_polygons_path = r'./gen_half_polygons'
  rename_img_path = r'./Train'

  ori_masks =r"./gen_half_polygons"
  save_half_regions =r"./gen_half_regions"

  MSRA_TD500(gt_path, img_path, save_four_gt)

  generate_half_polygons(gts_path, rename_img_path, save_half_polygons_path)  

  generate_half_regions(ori_masks, save_half_regions)



