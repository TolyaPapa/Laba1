from PIL import Image
from scipy.stats import skew, kurtosis,laplace
from scipy.optimize import curve_fit,minimize
from scipy.special import factorial, loggamma
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import sys
import cv2
import statistics as stt
import matplotlib.pyplot as plt
import glob


def getRed(redVal):
    return '#%02x%02x%02x' % (redVal, 0, 0)


def getGreen(greenVal):
    return '#%02x%02x%02x' % (0, greenVal, 0)


def getBlue(blueVal):
    return '#%02x%02x%02x' % (0, 0, blueVal)

def func_laplace(x,loc,scale):
    return    np.exp(-abs(x - loc) / scale) / (2. * scale)
def func_gaussian(x,mu,sigma):
    return    1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
def func_poisson(k, lamb):
    return (lamb ** k / factorial(k)) * np.exp(-lamb)


# Main Function solve problem
def Solve():
    g = open('OutPut.txt', "w")
    pp = PdfPages("AllHistogram.pdf")
    np.seterr(divide='ignore', invalid='ignore')
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore") #ignore some warnings from system
    cnt=0 #count how many images had taken
    for filename in glob.glob('D:/10semester/Progonov/Лабораторные работы/mirflickr/*.jpg'):
        photo = Image.open(filename)
        photo = photo.convert('RGB')
        g.write('Output for Image number {}\n\n'.format(cnt + 1))
        Red = []
        Green = []
        Blue = []

        width, height = photo.size  # define W and H
        for y in range(0, height):  # each pixel has coordinates
            for x in range(0, width):
                RGB = photo.getpixel((x, y))
                R, G, B = RGB  # now we can use the RGB value
                Red.append(R)
                Green.append(G)
                Blue.append(B)

        sorted(Red)
        sorted(Green)
        sorted(Blue)

        g.write('Max and min values of Red channel of image {} are: {}, {}\n'.format(cnt + 1, max(Red), min(Red)))
        g.write('Max and min values of Green channel of image {} are: {}, {}\n'.format(cnt + 1, max(Green), min(Green)))
        g.write('Max and min values of Blue channel of image {} are: {}, {}\n\n'.format(cnt + 1, max(Blue), min(Blue)))

        RedDict = dict((x, Red.count(x)) for x in set(Red))
        GreenDict = dict((x, Green.count(x)) for x in set(Green))
        BlueDict = dict((x, Blue.count(x)) for x in set(Blue))
        # ================================================
        # for Red channel

        g.write('Sum of Red channel is : {}\n'.format(sum(Red)))
        g.write('Median of Red channel is : {}\n'.format(stt.median(Red)))
        g.write('Lower and Upper quantile of Red channel are : {} {}\n'.format(np.quantile(Red, 0.25),
                                                                               np.quantile(Red, 0.75)))
        g.write('Mean value is : {}\n'.format(stt.mean(Red)))
        g.write('Skewness and Kurtosis are : {} {}\n'.format(skew(np.array(Red)), kurtosis(Red)))
        g.write('Average value of Red channel is : {}\n'.format(sum(Red) / (width * height)))
        # Find the Variance of Channel
        TempSum = 0
        for x in RedDict.keys():
            TempSum += x ** 2 * RedDict[x]
        TempSum /= (width * height)
        D = TempSum - stt.mean(Red) ** 2
        g.write('The Variance of Red channel is :  {}\n\n'.format(D))
        # ================================================
        # for Green channel

        g.write('Sum of Green channel is : {} \n'.format(sum(Green)))
        g.write('Median of Green channel is : {}\n'.format(stt.median(Green)))
        g.write('Lower and Upper quantile of Green channel are : {} {}\n'.format(np.quantile(Green, 0.25),
                                                                                 np.quantile(Green, 0.75)))
        g.write('Mean value is : {}\n'.format(stt.mean(Green)))
        g.write('Skewness and Kurtosis are : {} {}\n'.format(skew(np.array(Green)), kurtosis(Green)))
        g.write('Average value of Red channel is : {}\n'.format(sum(Green) / (width * height)))
        # Find the Variance of Channel
        TempSum = 0
        for x in GreenDict.keys():
            TempSum += x ** 2 * GreenDict[x]
        TempSum /= (width * height)
        D = TempSum - stt.mean(Green) ** 2
        testt=D
        g.write('The Variance of Green channel is :  {}\n\n'.format(D))
        # =====================================================
        # for Blue channel

        g.write('Sum of Blue channel is : {}\n'.format(sum(Blue)))
        g.write('Median of Blue channel is : {}\n'.format(stt.median(Blue)))
        g.write('Lower and Upper quantile of Blue channel are : {} {}\n'.format(np.quantile(Blue, 0.25),
                                                                                np.quantile(Blue, 0.75)))
        g.write('Mean value is : {}\n'.format(stt.mean(Blue)))
        g.write('Skewness and Kurtosis are : {} {}\n'.format(skew(np.array(Blue)), kurtosis(Blue)))
        g.write('Average value of Blue channel is : {}\n'.format(sum(Blue) / (width * height)))
        # Find the Variance of Channel
        TempSum = 0
        for x in BlueDict.keys():
            TempSum += x ** 2 * BlueDict[x]
        TempSum /= (width * height)
        D = TempSum - stt.mean(Blue) ** 2
        g.write('The Variance of Blue channel is :  {}\n\n'.format(D))

        histogram = photo.histogram()
        # Take only the Red counts
        l1 = histogram[0:256]
        summ=sum(l1)
        l1[:]=[x/summ for x in l1]

        # Take only the Green counts
        l2 = histogram[256:512]
        summ=sum(l2)
        l2[:]=[x/summ for x in l2]

        # Take only the Blue counts
        l3 = histogram[512:768]
        summ=sum(l3)
        l3[:]=[x/summ for x in l3]

        fig = plt.figure(figsize=(15, 8))

        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.set(xlabel="Red channel")
        ax2.set(title="Histogram of Image number " + str(cnt + 1), xlabel="Green channel")
        ax3.set(xlabel="Blue channel")

        # R histogram
        for i in range(0, 256):
            ax1.bar(i, l1[i], color=getRed(i), edgecolor=getRed(i), alpha=0.3)

        # G histogram

        for i in range(0, 256):
            ax2.bar(i, l2[i], color=getGreen(i), edgecolor=getGreen(i), alpha=0.3)

        # B histogram

        for i in range(0, 256):
            ax3.bar(i, l3[i], color=getBlue(i), edgecolor=getBlue(i), alpha=0.3)

        #=============================================
        x=np.linspace(0,255,256)
        #Red channel
        # fit histogram with gaussian ditribution
        popt, pcov = curve_fit(func_gaussian, x, l1)
        ax1.plot(x, func_gaussian(x, *popt), 'r-', label='fit:')

        # fit histogram with laplace distribution
        popt, pcov = curve_fit(func_laplace, x, l1)
        ax1.plot(x, func_laplace(x, *popt), 'r-', linestyle="--", label='fit:')

        # fit histogram with poisson distribution

        popt, pcov = curve_fit(func_poisson, x, l1)
        #ax1.plot(x, func_poisson(x, popt), 'r-', lw=2)

        ax1.legend(('Gaussian distribution', 'Laplace distribution'), loc='upper right')

        #Green channel
        #fit histogram with gaussian ditribution
        popt, pcov = curve_fit(func_gaussian, x, l2)
        ax2.plot(x, func_gaussian(x, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

        #fit histogram with laplace distribution
        popt, pcov = curve_fit(func_laplace, x, l2)
        ax2.plot(x, func_laplace(x, *popt), 'r-',linestyle = "--",label = 'fit:')

        # fit histogram with poisson distribution

        popt, pcov = curve_fit(func_poisson, x, l2)
        #ax2.plot(x, func_poisson(x,popt), 'r-', lw=2)

        ax2.legend(('Gaussian distribution', 'Laplace distribution'),loc='upper right')

        #Blue channel
        # fit histogram with gaussian ditribution
        popt, pcov = curve_fit(func_gaussian, x, l3)
        ax3.plot(x, func_gaussian(x, *popt), 'r-', label='fit1')

        # fit histogram with laplace distribution
        popt, pcov = curve_fit(func_laplace, x, l3)
        ax3.plot(x, func_laplace(x, *popt), 'r-', linestyle="--", label='fit')

        # fit histogram with poisson distribution

        popt, pcov = curve_fit(func_poisson, x, l3)
        ax3.plot(x, func_poisson(x,*popt), 'r-', lw=2, linestyle = ':')

        ax3.legend(('Gaussian distribution', 'Laplace distribution','Poisson distribution'),loc = 'upper right')

        pp.savefig()
        plt.close('all')
        photo.close()
        g.write('=================================================\n\n')
        cnt+=1
        if cnt>=1:
            break
    pp.close()
    g.close()
Solve()
