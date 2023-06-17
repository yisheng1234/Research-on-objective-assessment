
import cv2
import math
import numpy as numpy
import os


#定义最大灰度级数
gray_level =128

def maxGrayLevel(img):
    max_gray_level=0
    (height,width)=img.shape
    # print ("图像的高宽分别为：height,width",height,width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    # print("max_gray_level:",max_gray_level)
    return max_gray_level+1

def getGlcm(input,d_x,d_y):
    srcdata=input.copy()
    ret=[[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height,width) = input.shape

    max_gray_level=maxGrayLevel(input)
    #若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，标准化，无量纲化
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i]*gray_level / max_gray_level

    for j in range(height):
        for i in range(width):
            rows = srcdata[j][i]
            if(i+d_x<width):
                cols = srcdata[j + d_y][i+d_x]
                ret[rows][cols] += 1.0

            if(i-d_x>=0):
                cols = srcdata[j + d_y][i - d_x]
                ret[rows][cols] += 1.0



    #类似于归一化，便于计算
    # for i in range(gray_level):
    #     for j in range(gray_level):
    #         ret[i][j]/=float(height*width)

    return ret

def feature_computer(p):
    #con:对比度反应了图像的清晰度和纹理的沟纹深浅。纹理越清晰反差越大对比度也就越大。
    #asm:角二阶矩（能量），图像灰度分布均匀程度和纹理粗细的度量。当图像纹理均一规则时，能量值较大；反之灰度共生矩阵的元素值相近，能量值较小。
    #hom:同质性,反差分矩阵又称逆方差，反映了纹理的清晰程度和规则程度，纹理清晰、规律性较强、易于描述的，值较大。
    Con=0.0
    Asm=0.0
    Hom=0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con+=(i-j)*(i-j)*p[i][j]
            # print("i-j=============",i-j)
            # print("pij===========",p[i][j])
            Asm+=p[i][j]*p[i][j]
            Hom+=p[i][j]/(1+(i-j)*(i-j))

    return Asm,Con,Hom



def getfeature(img):
    try:
        img_shape=img.shape
    except:
        print ('imread error')
        return
    # img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 水平相邻1
    glcm_0=getGlcm(img, 1,0)
    glcm_0=numpy.array(glcm_0)
    # cv2.imshow("1",glcm_0)
    # cv2.imwrite("glcm.jpg",glcm_0*255)
    # cv2.waitKey()
    glcm_p=numpy.full_like(glcm_0,1)
    glcm_sum=0.0

    for i in range(gray_level):
        for j in range(gray_level):
            glcm_sum+=glcm_0[i][j]

    for i in range(gray_level):
        for j in range(gray_level):
            glcm_p[i][j]=glcm_0[i][j]/glcm_sum

    asm,con,hom=feature_computer(glcm_p)

    return [asm,con,hom]

