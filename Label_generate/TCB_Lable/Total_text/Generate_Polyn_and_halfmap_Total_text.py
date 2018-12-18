# -*- coding: utf-8 -*-
import numpy as np
import cv2, os

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
    # print gtlines[i]
    lst = gtlines[i].split(",")
    x_len = len(lst[0][5:])
    # print lst[0], lst[1]
    y_len = len(lst[1][6:])
  
    xs = lst[0][5:][:x_len-2].split(" ")
    ys = lst[1][6:][:y_len-2].split(" ")

    temp_x=[]
    for x in xs:
      if x!='':
        temp_x.append(x)      
    all_xs.append(temp_x)
    temp_y = []
    for y in ys:
      if y!='':
        temp_y.append(y) 
    all_ys.append(temp_y)
    # print len(all_xs), len(all_ys)
  
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
  max_nmus =[]
  for gt in os.listdir(gts_path):
    if endWith(gt, '.txt'):
      gtfile = open(gts_path + "/" + gt, 'r')
      save_map=save_half_polygons_path + r"/" +gt[8:-4]+".tif"
      save_img=save_half_polygons_path + r"/" +gt[8:-4]+".jpg"
      img_file = img_path + "/" + gt[8:-4]+".jpg"  
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

if __name__ == "__main__":

  gts_path = r'./Train_gt'
  save_half_polygons_path = r'./gen_half_polygons'
  img_path = r'./Train'

  ori_masks =r"./gen_half_polygons"
  save_half_regions =r"./gen_half_regions"

  if not os.path.exists(save_half_regions):
    os.makedirs(save_half_regions)  

  if not os.path.exists(save_half_polygons_path):
    os.makedirs(save_half_polygons_path)

  generate_half_polygons(gts_path, img_path, save_half_polygons_path)  

  generate_half_regions(ori_masks, save_half_regions)



