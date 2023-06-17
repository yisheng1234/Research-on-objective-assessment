# 旋转不变等价模式
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
def cal_basic_lbp(img,i,j):#比中心像素大的点赋值为1，比中心像素小的赋值为0，返回得到的二进制序列
    sum = []
    if img[i - 1, j ] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i - 1, j+1 ] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i , j + 1] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i + 1, j+1 ] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i + 1, j ] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i + 1, j - 1] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i , j - 1] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    if img[i - 1, j - 1] > img[i, j]:
        sum.append(1)
    else:
        sum.append(0)
    return sum
def lbp_uniform(img):
    revolve_array = np.zeros(img.shape,np.uint8)
    width = img.shape[0]
    height = img.shape[1]
    for i in range(1,width-1):
        for j in range(1,height-1):
            sum_ = cal_basic_lbp(img,i,j) #获得二进制
            num_ = calc_sum(sum_)  #获得跳变次数
            if num_ <= 2:
                revolve_array[i,j] = count(bin_to_decimal(sum_)) #若跳变次数小于等于2，则将该二进制序列1的位数作为LBP值
            else:
                revolve_array[i,j] = 9  # P + 1 = 8 + 1 = 9
    return revolve_array
def calc_sum(r):  # 获取值r的二进制中跳变次数
    sum_ = 0
    for i in range(0,len(r)-1):
        if(r[i] != r[i+1]):
            sum_ += 1
    return sum_
def show_uniform_hist(img_array):
    return show_hist(img_array, [9], [0,9])
def show_hist(img_array,im_bins,im_range):
    # print(img_array.shape)
    hist = cv.calcHist([img_array], [0], None, im_bins, im_range)
    hist = cv.normalize(hist, hist).flatten()
    # width = 0.3
    # # 创建等差数列
    # ind = np.linspace(0.5, 8.5, 9)
    # # make a square figure
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # # Bar Plot
    # ax.bar(ind, hist, width, color='green')
    # # Set the ticks on x-axis
    # ax.set_xticks(ind)
    # ax.set_xticklabels(['0','1','2','3','4','5','6','7','8'])
    # # labels
    # # ax.set_xlabel('Country')
    # # ax.set_ylabel('GDP (Billion US dollar)')
    # # title
    # # ax.set_title('Top 10 GDP Countries', bbox={'facecolor': '0.8', 'pad': 5})
    # plt.grid(True)
    # plt.show()
    # # plt.savefig("bar.jpg")
    # # plt.close()
    #
    #
    # # print(hist)
    # # plt.plot(hist, color='r')
    # # plt.xlim(im_range)
    # # plt.show()
    return hist


def count(num):
    cnt = 0
    while num:
        if num & 1 == 1:
            cnt += 1
        num = num >> 1
    return cnt
def bin_to_decimal(bin):#二进制转十进制
    res = 0
    bit_num = 0 #左移位数
    for i in bin[::-1]:
        res += i << bit_num   # 左移n位相当于乘以2的n次方
        bit_num += 1
    return res
def getFeature(img1):
    # img1=cv.imread("dataset/111.jpg")
    # img1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    uniform_array = lbp_uniform(img1)
    hist=show_uniform_hist(uniform_array)
    return hist
# print(hist)
# print(hist.shape)
# plt.imshow(uniform_array,cmap='Greys_r')
# plt.show()
