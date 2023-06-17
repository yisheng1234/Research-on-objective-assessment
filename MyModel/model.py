
import torch
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import Last_GLCM
from torch.utils.data import Dataset
from torchvision import transforms,models
import cv2
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
import numpy as np
from torch import nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
import CAM.useCam as cam
import RMSE
import  Last_cut1


from NRSL import LCN


class MyDataset(Dataset):
	def __init__(self, txt, type, transform=None, target_transform=None, loader=default_loader):
		fh = open(txt, 'r')
		imgs = []
		self.type = type
		for line in fh:
			line = line.rstrip()
			words = line.split()  # 分割成文件名和标签
			imgs.append((words[0], float(words[1])))
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader
	def __getitem__(self, index):
		fn, label = self.imgs[index]
		img=cv2.imread(fn)
		if self.transform is not None:
			img = self.transform(img)
		return img, label
	def __len__(self):
		return len(self.imgs)
tran_Totensor = transforms.ToTensor()
tran_compose = transforms.Compose([tran_Totensor])
train_data = MyDataset(txt='dataset/train1.txt', type="train", transform=None)
test_data = MyDataset(txt='dataset/test1.txt', type="test", transform=None)

test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False, num_workers=0, drop_last=False)
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=False, num_workers=0, drop_last=False)


mobileNetV3=models.mobilenet_v3_small(pretrained=True)
mobileNetV3.classifier=nn.Sequential(nn.Linear(576,150))
# print(mobileNetV3)
save_model=torch.load("model/mobileV3_150.pth")
model_dict = mobileNetV3.state_dict()
state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
model_dict.update(state_dict)
mobileNetV3.load_state_dict(model_dict)
# print(mobileNetV3)
train_X=np.zeros((184, 312), dtype=np.float32)
train_Y=np.zeros((184),dtype=np.float32)
mobileNetV3.eval()
for j,data in enumerate(train_loader,0):
	imgs, targets = data
	# print(imgs.__class__)
	# print(imgs)
	targets = targets.flatten().numpy()
	train_Y[8 * j:8 * (j + 1)] = targets
	s=0
	for img in imgs:
		img = img.numpy()
		# 清晰度特征    144
		gray_cam,color_cam,visualization=cam.getHot(img)
		define_feature = Last_cut1.cut(img,gray_cam)
		# GLCM特征提取
		# 对比度归一化图像
		img_lcn = LCN.getLCN(img)
		img_lcn = LCN.imshow(img_lcn)
		img_lcn = np.asarray(img_lcn)
		img_shape = img_lcn.shape
		h, w, c = img_shape
		img_down2 = cv2.pyrDown(img_lcn, (h / 2, w / 2))
		b = img_lcn[:, :, 0]
		g = img_lcn[:, :, 1]
		r = img_lcn[:, :, 2]

		img_down2_r=img_down2[:, :, 0]
		img_down2_g=img_down2[:, :, 1]
		img_down2_b=img_down2[:, :, 2]
		glcm_feature=np.zeros(18)
		glcm_feature[0],glcm_feature[1],glcm_feature[2]=Last_GLCM.getfeature(r)
		glcm_feature[3],glcm_feature[4],glcm_feature[5]=Last_GLCM.getfeature(g)
		glcm_feature[6],glcm_feature[7],glcm_feature[8]=Last_GLCM.getfeature(b)
		glcm_feature[9], glcm_feature[10], glcm_feature[11] = Last_GLCM.getfeature(img_down2_r)
		glcm_feature[12], glcm_feature[13], glcm_feature[14] = Last_GLCM.getfeature(img_down2_g)
		glcm_feature[15], glcm_feature[16], glcm_feature[17] = Last_GLCM.getfeature(img_down2_b)
		scaler = StandardScaler()
		glcm_feature = glcm_feature.reshape(-1, 1)
		glcm_feature = scaler.fit_transform(glcm_feature)
		glcm_feature.reshape(18)



		# 深度特征
		tensor_img=tran_compose(img)
		# print(tensor_img.shape)
		deep_feature=mobileNetV3(tensor_img.reshape(1, 3, 240, 320))
		deep_feature = deep_feature.detach().numpy()
		deep_feature = deep_feature.flatten()


		# 特征融合
		all_feature=np.append(np.append(deep_feature,define_feature),glcm_feature)
		# all_feature=np.append(glcm_feature,define_feature)
		# all_feature=define_feature
		all_feature=all_feature.reshape(-1,1)
		all_feature=scaler.fit_transform(all_feature)
		# print(all_feature.shape)
		all_feature=all_feature.reshape(312)
		train_X[s+j*8,:]=all_feature
		s+=1

# clf = SVR(C=1500, cache_size=1000, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
# 			kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=True)
clf = SVR(C=1500, cache_size=1000, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
			kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=True)

clf.fit(train_X,train_Y)

test_X=np.zeros((48, 312), dtype=np.float32)
test_Y=np.zeros((48),dtype=np.float32)
for t,data in enumerate(test_loader,0):
	imgs, targets = data

	targets = targets.flatten().numpy()
	test_Y[8 * t:8 * (t + 1)] = targets
	s=0
	for img in imgs:
		img=img.numpy()

		# 清晰度特征    144
		gray_cam, color_cam, OriginalandCam = cam.getHot(img)
		define_feature = Last_cut1.cut(img, gray_cam)

		# GLCM特征提取
		# 对比度归一化图像
		img_lcn = LCN.getLCN(img)
		img_lcn = LCN.imshow(img_lcn)
		img_lcn = np.asarray(img_lcn)
		img_shape = img_lcn.shape
		h, w, c = img_shape
		# print(h, w, c)
		img_down2 = cv2.pyrDown(img_lcn, (h / 2, w / 2))

		b = img_lcn[:, :, 0]
		g = img_lcn[:, :, 1]
		r = img_lcn[:, :, 2]
		# print(img.shape)
		img_down2_r = img_down2[:, :, 0]
		img_down2_g = img_down2[:, :, 1]
		img_down2_b = img_down2[:, :, 2]

		glcm_feature=np.zeros(18)
		glcm_feature[0],glcm_feature[1],glcm_feature[2]=Last_GLCM.getfeature(r)
		glcm_feature[3],glcm_feature[4],glcm_feature[5]=Last_GLCM.getfeature(g)
		glcm_feature[6],glcm_feature[7],glcm_feature[8]=Last_GLCM.getfeature(b)
		glcm_feature[9], glcm_feature[10], glcm_feature[11] = Last_GLCM.getfeature(img_down2_r)
		glcm_feature[12], glcm_feature[13], glcm_feature[14] = Last_GLCM.getfeature(img_down2_g)
		glcm_feature[15], glcm_feature[16], glcm_feature[17] = Last_GLCM.getfeature(img_down2_b)
		scaler = StandardScaler()
		glcm_feature=glcm_feature.reshape(-1,1)
		glcm_feature=scaler.fit_transform(glcm_feature)
		glcm_feature.reshape(18)

		# 深度特征
		tensor_img=tran_compose(img)
		deep_feature=mobileNetV3(tensor_img.reshape(1, 3, 240, 320))
		deep_feature = deep_feature.detach().numpy()
		deep_feature = deep_feature.flatten()
# 		#
# 		#
# 		# # 特征融合
# 		all_feature=np.append(glcm_feature,define_feature)
		all_feature=np.append(np.append(deep_feature,define_feature),glcm_feature)
# 		all_feature=define_feature
		all_feature=all_feature.reshape(-1,1)

		all_feature=scaler.fit_transform(all_feature)
		all_feature=all_feature.reshape(312)
		test_X[s+t*8,:]=all_feature
		s+=1
# for i in range(len(test_X)):
# 	sample = test_X[i]
# 	for j in range(len(sample)):
# 		if np.isnan(sample[j]):
# 			sample[j] = 0
predict=clf.predict(test_X)
print(predict)
print(test_Y)
print("得分:", r2_score(test_Y, predict))
print("总体PLCC=",stats.pearsonr(predict,test_Y)[0])
print("总体SROCC=",stats.spearmanr(predict,test_Y)[0])
print("总体rmse=",RMSE.get_rmse(test_Y,predict))

plt.plot(np.arange(1,49),predict,'.-',label="predict")
plt.plot(np.arange(1,49),test_Y,'.-',label="true")
plt.legend()
plt.show()










