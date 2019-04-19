from PIL import Image
from scipy.stats import skew, kurtosis
import os
import numpy as np
import statistics as stt
import glob

image_list = []
#Import 100 image from folder /mirflickr
def Input():
    cnt = 0  # Count how many image have been loaded
    for filename in glob.glob('D:/10semester/Progonov/Лабораторные работы/mirflickr/*.jpg'):
        # and cnt < 100:  # assuming jpg
        im = Image.open(filename)
        im = im.convert('RGB')
        image_list.append(im)
        #im.close()
        cnt += 1
        if cnt >= 100:
            break
Input()
#g.write(len(image_list)) #g.write out quantity of images in list

#Function solve problem
def Solve():
    g = open('OutPut.txt', "w")
    for i in range(2):
        g.write('Output for Image number {}\n'.format(i+1))
        Red = []
        Green = []
        Blue = []
        #photo = Image.open(image_list[i])  # your image
        # photo = photo.convert('RGB')
        photo = image_list[i]
        width = photo.size[0]  # define W and H
        height = photo.size[1]
        for y in range(0, height):  # each pixel has coordinates
            for x in range(0, width):
                RGB = photo.getpixel((x, y))
                R, G, B = RGB  # now we can use the RGB value
                Red.append(R)
                Green.append(G)
                Blue.append(B)
        for temp in (Red,Green,Blue):
            print('Max and min values of Red channel of image {} are: {}, {}\n'.format(i + 1, max(temp), min(temp)))
        sorted(Red)
        g.write('Max and min values of Red channel of image {} are: {}, {}\n'.format(i+1,max(Red),min(Red)) )
        g.write('Max and min values of Green channel of image {} are: {}, {}\n'.format(i + 1, max(Green), min(Green)))
        g.write('Max and min values of Blue channel of image {} are: {}, {}\n'.format(i + 1, max(Blue), min(Blue)))

        RedDict = dict((x, Red.count(x)) for x in set(Red))
        summ = 0
        for x in RedDict.keys():
            g.write('{} {}\n'.format(x, RedDict[x]))
            summ+=x
        g.write('Sum of Red channel is : {}\n'.format(summ))
        g.write('Median of Red channel is : {}\n'.format(stt.median(Red)))
        g.write('Mean value is : {}\n'.format(stt.mean(Red)))
        g.write('Skewness and Kurtosis are : {} {}\n'.format(skew(Red),kurtosis(Red)))
        g.write('{}\n'.format(sum(Red)/(width*height)))
        # Find the Variance of List
        TempSum = 0
        for x in RedDict.keys():
            TempSum += x ** 2 * RedDict[x]
        TempSum /=(width*height)
        D = TempSum - stt.mean(Red) ** 2
        g.write('The Variance is {}\n'.format(D))
        photo.close()
    g.close()

Solve()
