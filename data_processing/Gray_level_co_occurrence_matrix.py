import cv2
import math

oldPath = 'test/img'
newPath = 'test/new'
import glob
import os


def cut(path, i, newPath):
        image = cv2.imread(path)
        print(newPath + path[8:])
        cv2.imwrite(newPath+path[i:], image[image.shape[0] // 4: image.shape[0] // 4 * 3, image.shape[1] // 4: image.shape[1] // 4 * 3])



# imgpath = "./new" 
file_name = "test/test.csv"

# import glob
# import os

#定义最大灰度级数
gray_level = 16

def maxGrayLevel(img):
    max_gray_level=0
    (height,width)=img.shape
    print(height,width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level+1

def getGlcm(input,d_x,d_y):
    srcdata=input.copy()
    ret=[[0.0 for i in range(gray_level)] for j in range(gray_level)]   #建一个16*16且每个元素都是0.0的矩阵
    (height,width) = input.shape
    
    max_gray_level=maxGrayLevel(input)   #得到最大灰度级数
    
    #若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i]*gray_level / max_gray_level

    for j in range(height-d_y):
        for i in range(width-d_x):
             rows = srcdata[j][i]
             cols = srcdata[j + d_y][i+d_x]
             ret[rows][cols]+=1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j]/=float(height*width)

    return ret

def feature_computer(p):
    Con=0.0
    Eng=0.0
    Asm=0.0
    Idm=0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con+=(i-j)*(i-j)*p[i][j]
            Asm+=p[i][j]*p[i][j]
            Idm+=p[i][j]/(1+(i-j)*(i-j))
            if p[i][j]>0.0:
                Eng+=p[i][j]*math.log(p[i][j])
    return Asm,Con,-Eng,Idm

def test(image_name):
    img = cv2.imread(image_name)
    try:
        img_shape=img.shape
    except:
        print('imread error')
        return

    img=cv2.resize(img,(img_shape[1]//2,img_shape[0]//2),interpolation=cv2.INTER_CUBIC)

    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #glcm_0=getGlcm(img_gray, 1,0)
    #glcm_1=getGlcm(src_gray, 0,1)
    glcm_2=getGlcm(img_gray, 1,1)
    #glcm_3=getGlcm(src_gray, -1,1)

    asm,con,eng,idm=feature_computer(glcm_2)

    return [asm,con,eng,idm]


import csv
import codecs
def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    out = codecs.open(file_name,'a','utf-8')
    #设定写入模式
    csv_write = csv.writer(out,dialect='excel')
    #写入具体内容
    csv_write.writerow(datas)
    out.close()



    # file_csv = codecs.open(file_name,'w+','utf-8')#追加
    # writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    # for data in datas:
    #     writer.writerow(data)
    print("保存文件成功，处理结束")


def main():

    paths = glob.glob(os.path.join(oldPath, '*.jpg'))
    paths.sort()
    print(paths)

    for path in paths:
        cut(path, 8, newPath)

    imgpaths = glob.glob(os.path.join(newPath, '*.jpg'))
    imgpaths.sort()

    # title = ['Id', 'asm', 'con' ,'eng', 'idm', 'label']
    # data_write_csv(file_name,title)

    for imgpath in imgpaths:
        result = [str(item) for item in test(imgpath)]
        #print(result[0])
        result.insert(0, imgpath[9:-4])
        result.append(imgpath[9:12])
        print(result)
        data_write_csv(file_name, result)

    # result = [str(item) for item in test(imgpath)]
    # result.insert(0, imgpath[6:-4])
    # result.append('1.0')
    # data_write_csv(file_name, result)
main()