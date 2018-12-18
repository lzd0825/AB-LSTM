import numpy as np
import cv2

def RegionGrowing(I, init_pos, reg_maxdist, J):
	row = I.shape[0]
	col = I.shape[1]
	x0 = init_pos[0]
	y0 = init_pos[1]
	print x0, y0
	reg_mean = I[x0, y0]
	J[x0, y0] = 1
	reg_sum = reg_mean
	reg_num = 0
	count = 1
	reg_choose = np.zeros((I.shape[0]*I.shape[1], 2))
	reg_choose[reg_num, :] = init_pos
	
	reg_num = 1
	num = 1
	while count > 0:
		s_temp = 0
		count=0
		for k in range(num):
			i = int(reg_choose[reg_num - num + k, 0])
			j = int(reg_choose[reg_num - num + k, 1])
			print "i, j :", i, j			

			if J[i,j] == 1 and i>0 and i<row and j>0 and j<col:
				for u in range(-1,2):
					print u
					for v in range(-1, 2):
						if (J[i + u, j + v] == 0) and (abs(I[i+u, j+v] - reg_mean)<=reg_maxdist):
							J[i+u, j+v] = 1
							count = count+1
							reg_choose[reg_num+count, :]=[i+u, j+v]
							s_temp=s_temp+I[i+u, j+v]
		num =count
		reg_num=reg_num+count
		reg_sum=reg_sum+s_temp
		reg_mean=reg_sum/reg_num
	# cv2.imshow("J", J)
	# cv2.waitKey(0)

I = cv2.imread(r'./img_17.jpg', 0)
gtfile = open(r'./img_17.txt', 'r')

gtlines =  gtfile.readlines()
J = np.zeros((I.shape[0], I.shape[1]))

for line in gtlines:
	x,y=line.replace("\n","").split(', ')
	# 
	print x, y
	RegionGrowing(I, [int(y), int(x)], 10, J)

cv2.imshow("J", J)
cv2.waitKey(0)