import os

import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from scipy.signal import convolve2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def local_contrast_norm(image, radius=9):
	"""
	image: torch.Tensor , .shape => (1,channels,height,width)

	radius: Gaussian filter size (int), odd
	"""
	if radius % 2 == 0:  # LCN核的大小为奇数
		radius += 1

	def get_gaussian_filter(kernel_shape):
		x = np.zeros(kernel_shape, dtype='float64')

		# 二维高斯函数
		def gauss(x, y, sigma=2.0):
			Z = 2 * np.pi * sigma ** 2
			return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

		mid = np.floor(kernel_shape[-1] / 2.)  # 求出卷积核的中心位置(mid,mid)
		for kernel_idx in range(0, kernel_shape[1]):  # 遍历每一层
			for i in range(0, kernel_shape[2]):  # 遍历x轴
				for j in range(0, kernel_shape[3]):  # 遍历y轴
					x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)  # 计算出高斯权重

		return x / np.sum(x)

	n, c, h, w = image.shape[0], image.shape[1], image.shape[2], image.shape[3]  # (图片数、层数、x轴、y轴大小)

	gaussian_filter = torch.Tensor(get_gaussian_filter((1, c, radius, radius))).to(device)  # 创建卷积核
	filtered_out = F.conv2d(image, gaussian_filter, padding=radius - 1)  # 卷积 (∑ipq Wpq.X i,j+p,k+q)
	mid = int(np.floor(gaussian_filter.shape[2] / 2.))  # 获得卷积核的中心位置

	### Subtractive Normalization
	centered_image = image - filtered_out[:, :, mid:-mid, mid:-mid]  # Vijk
	# ↑由于padding为radius-1,filered_out比实际图片要大,截取mid:-mid后才是有效区域)

	## Variance Calc
	sum_sqr_image = F.conv2d(centered_image.pow(2), gaussian_filter, padding=radius - 1)  # ∑ipqWpq.v2 i,j+p,k+q
	s_deviation = sum_sqr_image[:, :, mid:-mid, mid:-mid].sqrt()  # σ jk
	per_img_mean = s_deviation.mean()  # c

	## Divisive Normalization
	divisor = np.maximum(per_img_mean.cpu().detach().numpy(), s_deviation.cpu().detach().numpy())  # max(c, σjk)
	divisor = np.maximum(divisor, 1e-4)
	new_image = centered_image / torch.Tensor(divisor).to(device)  # Yijk

	return new_image.to(device)


##############测试##########################

import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# 显示tensor图片
def imshow(tensor, title=None):
	image = tensor.cpu().clone()
	image = image.squeeze(0)
	unloader = transforms.ToPILImage()
	image = unloader(image)
	# plt.imshow(image)
	# if title is not None:
	# 	plt.title(title)
	# plt.show()
	return image
def getLCN(image):
	input_transform = transforms.Compose([
		transforms.ToTensor(),
	])

	# image = plt.imread("../dataset/train_image/3.jpg")

	img = torch.Tensor([np.array(image).transpose((2, 0, 1))]).to(device)
	img = local_contrast_norm(img, 7)

	img = img[0].cpu().numpy().transpose((1, 2, 0))
	img = img.astype(np.uint8)
	img = input_transform(img)
	return img
	# imshow(img)

if __name__ == "__main__":
	input_transform = transforms.Compose([
	    transforms.ToTensor(),
	])

	image = plt.imread("../dataset/train_image/3.jpg")

	img = torch.Tensor([np.array(image).transpose((2, 0, 1))]).to(device)
	img = local_contrast_norm(img, 7)

	img = img[0].cpu().numpy().transpose((1, 2, 0))
	img = img.astype(np.uint8)
	img = input_transform(img)
	imshow(img)
