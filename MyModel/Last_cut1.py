from PIL import Image
import numpy as np
import cv2
import LBP as lbp
def cut(img,cam):

	# size = img.size
	H,W,C=img.shape
	# 准备将图片切割成100张小图片
	weight_num=8
	height_num=2
	weight = int(W // weight_num)
	height = int(H //height_num)
	# 80 * 80
	# 将BGR转换为RGB
	img= img[:, :, ::-1]
	#
	img=Image.fromarray(np.uint8(img))
	cam=Image.fromarray(cam)

	feature=np.empty(9*height_num*weight_num)
	for j in range(height_num):
		for i in range(weight_num):
			box = (weight * i, height * j, weight * (i + 1), height * (j + 1))
			region = img.crop(box)
			region_cam=cam.crop(box)
			# 转换为灰度图
			region=region.convert('L')
			# 转为ndarray
			np_region=np.array(region)
			np_region_cam=np.array(region_cam)
			h,w=np_region_cam.shape
			sum=0
			for k in range(h):
				for s in range(w):
					sum+=np_region_cam[k,s]

			wight=sum/((h*w)/10)
			hist=lbp.getFeature(np_region)
			feature_region=wight*hist
			# feature_region=hist
			feature[(weight_num*j+i)*9:(weight_num*j+i+1)*9]=feature_region
	return feature
