import numpy as np
import copy
import cv2

def plot_rects(img, region_block_yxs_ori, region_block_yxs_mer, txt_file):
	det_img = copy.copy(img)
	for i in range(len(region_block_yxs_mer)):
		y1 = min(region_block_yxs_mer[i][:, 0])
		y2 = max(region_block_yxs_mer[i][:, 0])
		x1 = min(region_block_yxs_mer[i][:, 1])
		x2 = max(region_block_yxs_mer[i][:, 1])
		y01 = min(region_block_yxs_ori[i][:, 0])
		y02 = max(region_block_yxs_ori[i][:, 0])
		x01 = min(region_block_yxs_ori[i][:, 1])
		x02 = max(region_block_yxs_ori[i][:, 1])

		if (y02 - y01) <= 0.5 * (y2 - y1):
			coords = region_block_yxs_ori[i]
			rect = cv2.minAreaRect(coords)
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)
			c_box = np.zeros(box.shape, dtype=np.uint64)
			c_box[:, 1] = box[:, 0] 
			c_box[:, 0] = box[:, 1]
			cv2.drawContours(det_img, [c_box], 0, (0, 0, 255), 5)	
		else:
			mask1 = cv2.rectangle(det_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
			txt_record = str(x1)+','+ str(y1)+','+str(x2)+','+str(y2)		
		
		win = cv2.namedWindow("det_img", flags = 0)
		cv2.imshow("det_img", det_img)
		cv2.waitKey(0)
		# txt_file.write(txt_record+"\n")
	return det_img
