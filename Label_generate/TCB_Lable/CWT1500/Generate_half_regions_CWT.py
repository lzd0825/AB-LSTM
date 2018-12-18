# -*- coding: utf-8 -*-
import cv2, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from math import sqrt
from scipy import optimize
import random

# 生成上下14个顶点，保存在一个txt文件中，并且在原图上进行可视化
def save_draw(img, save_file, center_file, cdets):
  for i in xrange(len(cdets)):
    bbox= cdets[i][:4]
    info_bbox = cdets[i][4:32]
    pts = [info_bbox[i] for i in xrange(28)]
    temp=''
    for p in xrange(0,28,2):
      x0 = int(bbox[0]) + int(pts[p%28])
      y0= int(bbox[1]) + int(pts[(p+1)%28])
      print "x0, y0 >>>>>: ",x0, y0
      if temp =='':
        temp=str(x0)+','+str(y0)
      else:
        temp=temp+','+str(x0)+','+str(y0)  

      # cv2.circle(img,(bbox[0], int(bbox[1])),5,(0,255,255),1) 
      # cv2.circle(img,(bbox[2], int(bbox[3])),5,(255,255,0),1)      
      # cv2.circle(img,(int(bbox[0]) + int(pts[p%28]), int(bbox[1]) + int(pts[(p+1)%28])), 30, (0, 255,0), -1)        
      cv2.line(img,(int(bbox[0]) + int(pts[p%28]), int(bbox[1]) + int(pts[(p+1)%28])), (int(bbox[0]) + int(pts[(p+2)%28]), int(bbox[1]) + int(pts[(p+3)%28])),(255,0,0),10) 
      # win1=cv2.namedWindow('winname', flags=0)
      # cv2.imshow("winname", img)
      # cv2.waitKey(0)
    # print img
    cv2.imwrite(save_file, img)
    center_file.write(temp+"\n")
    print "temp >>> ", temp
def endWith(s, *endstring):
   array = map(s.endswith, endstring)
   if True in array:
      return True
   else:
      return False 

def generate_new_gt_img(gt_path, img_path, save_new_gt_img):
  for gt in os.listdir(gt_path):
      if endWith(gt, '.txt'):
          gtfile = open(gt_path + "/" + gt, 'r')
          center_file = open(save_new_gt_img + "/" + gt[:-4]+'.gt', 'w')

          gtlines = gtfile.readlines()
          gtlines_len = len(gtlines)
          print gtlines
          imf=img_path + "/" +gt[:-4]+".jpg"

          save_file=save_new_gt_img + "/" +gt[:-4]+".png"
          img = cv2.imread(imf)

          cdets=[]
          for i in range(gtlines_len):
              x0,y0,x1,y1,w0,h0,w1,h1,w2,h2,w3,h3,w4,h4,w5,h5,w6,h6,w7,h7,w8,h8,w9,h9,w10,h10,w11,h11,w12,h12,w13,h13 = gtlines[i].split(",")
              cdet =[int(x0),int(y0),int(x1),int(y1),int(w0),int(h0),int(w1),int(h1),int(w2),int(h2),int(w3),int(h3),int(w4),int(h4),int(w5),int(h5),int(w6),int(h6),int(w7),int(h7),int(w8),int(h8),int(w9),int(h9),int(w10),int(h10),int(w11),int(h11),int(w12),int(h12),int(w13),int(h13)]
              cdets.insert(-1,cdet)
          save_draw(img, save_file,center_file, cdets)
          gtfile.close()
          center_file.close()
      else:
          continue
      print "Next"
  print "over all"


# 根据生成上下14个顶点，生成多边形
def draw_counter(lst, Half_map, h, w, cr1, cr2, flag):    
    ups=[]
    downs=[]
    for i in range(0, 14, 2):
      x1=lst[i]; y1=lst[i+1]   
      x2=lst[27-i-1]; y2=lst[27-i]

      lambda1=1*1.0/3; lambda2=3
      x01 = int((x1+lambda2*x2)/(1+lambda2)); y01 = int((y1+lambda2*y2)/(1+ lambda2))
      x02 = int((x1+lambda1*x2)/(1+lambda1)); y02 = int((y1+lambda1*y2)/(1+lambda1))
      if flag==1:
        cv2.circle(Half_map, (int(x01), int(y01)), 30, cr1, -1)
        cv2.circle(Half_map, (int(x02), int(y02)), 30, cr1, -1)

      temp_ups = [x01, y01]
      temp_downs = [x02, y02]
      ups.append(temp_ups)
      downs.append(temp_downs)
    # # Draw ups line 

    for i in range(len(ups)-1):
      if flag == 1:
        cv2.line(Half_map,(ups[i][0], ups[i][1]),(downs[i][0], downs[i][1]), cr2,10)
      cv2.line(Half_map,(ups[i][0], ups[i][1]),(ups[i+1][0], ups[i+1][1]),cr2,10)
        
    # Draw bottom line 
    for i in range(len(downs)-1):
        cv2.line(Half_map,(downs[i][0], downs[i][1]),(downs[i+1][0], downs[i+1][1]),cr2,10)

    # Draw two endpoints line
    cv2.line(Half_map,(downs[0][0], downs[0][1]),(ups[0][0], ups[0][1]),cr2,10)
    cv2.line(Half_map,(downs[6][0], downs[6][1]),(ups[6][0], ups[6][1]),cr2,10)

def endWith(s, *endstring):
   array = map(s.endswith, endstring)
   if True in array:
      return True
   else:
      return False 

def generate_half_maps(gts_path,save_half_map,img_path):
  for gt in os.listdir(gts_path):
      if endWith(gt, '.gt'):
          gtfile = open(gts_path + "/" + gt, 'r')
          print gtfile
          save_map=save_half_map + r"/" +gt[:-3]+".tif"
          save_img=save_half_map + r"/" +gt[:-3]+"_img.bmp"
          img_file = img_path + "/" + gt[:-3]+".jpg"        
          img = cv2.imread(img_file)
          h, w, _ = img.shape
          Half_map = np.zeros((img.shape[0], img.shape[1]))

          gtlines = gtfile.readlines()
          gtlines_len = len(gtlines) 
          for i in range(gtlines_len):
              x_up1, y_up1, x_up2, y_up2, x_up3, y_up3, x_up4, y_up4, x_up5, y_up5, x_up6, y_up6, x_up7, y_up7, x_dw1, y_dw1, x_dw2, y_dw2, x_dw3, y_dw3, x_dw4, y_dw4, x_dw5, y_dw5, x_dw6, y_dw6, x_dw7, y_dw7=gtlines[i].split(",")
              lst=[int(x_up1), int(y_up1), int(x_up2), int(y_up2), int(x_up3), int(y_up3), int(x_up4), int(y_up4), int(x_up5), int(y_up5), int(x_up6), int(y_up6), int(x_up7), int(y_up7), int(x_dw1), int(y_dw1), int(x_dw2), int(y_dw2), int(x_dw3), int(y_dw3), int(x_dw4), int(y_dw4), int(x_dw5), int(y_dw5), int(x_dw6), int(y_dw6), int(x_dw7), int(y_dw7)]

              draw_counter(lst, img, h, w,(0,255,0), (0,0,255), 1)
              draw_counter(lst, Half_map, h, w,(255,255,255), (255,255,255), 0)
          cv2.imwrite(save_img, img)
          cv2.imwrite(save_map, Half_map)

      gtfile.close()
      print "Next one"
  print "All Done"


def generate_half_regions(save_half_map, save_mask):

  masks = [flg for flg in os.listdir(save_half_map) if '.tif' in flg or '.tif' in flg]
  for msk in masks:     
    msk_file = save_half_map+'/'+msk
    save_file = save_half_regions+'/'+msk[:-8]+'.png'
    I =cv2.imread(r'./train'+'/'+msk[:-8]+'.jpg')
    img=cv2.imread(msk_file)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,80, 255, cv2.THRESH_BINARY)
    # 识别轮廓
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img = np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8)
    rr=[(0,255,255),(0,0,255)]
    i=0
    for cnt in contours:
      r = (random.randint(0,250),random.randint(50,150),random.randint(100,200))
      # r = (0,255,255) 
      print r 
      # 近似多边形，True表示闭合
      approx = cv2.approxPolyDP(cnt, 1, True)   
      hull = cv2.convexHull(cnt)
      cv2.drawContours(img, [approx], -1, (255,255,255), -1)
      i=i+1 
    cv2.imwrite(save_file, img)
    print "Next one"
  print "All Done"

if __name__ == "__main__":

  gt_path=r"./train_gt"
  img_path=r"./train"
  save_new_gt_img=r"./save_new_gt_img"

  gts_path = r'./save_new_gt_img'
  save_half_map = r'./save_half_map'

  save_half_regions =r"./save_half_regions"
  if not os.path.exists(save_half_regions):
    os.makedirs(save_half_regions)


  if not os.path.exists(save_half_map):
     os.makedirs(save_half_map)


  if not os.path.exists(save_new_gt_img):
     os.makedirs(save_new_gt_img)

  generate_new_gt_img(gt_path, img_path, save_new_gt_img)

  generate_half_maps(gts_path,save_half_map,img_path)

  generate_half_regions(save_half_map, save_half_regions )

      

