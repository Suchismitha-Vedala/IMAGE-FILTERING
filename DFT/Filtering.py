# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv
import cv2
import sys
import numpy as np
from numpy.random import rand
from datetime import datetime
from numpy import zeros, newaxis
from matplotlib import pyplot as plt
import math,cmath
from PIL import Image

class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        
       
        if filter_name == 'ideal_l':
            self.filter = 'ideal_l' #self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = 'ideal_h' #self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = 'butterworth_l' #self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = 'butterworth_h' #self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = 'gaussian_l' #self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = 'gaussian_h' #self.get_gaussian_high_pass_filter
       
       
        self.cutoff = cutoff
        self.order = order


    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""
        M,N=shape
        H=np.zeros((int(M),int(N)),np.float)
        for u in range(M):
            for v in range(N):
                value = math.sqrt((u-(M/2))**2 +(v-(N/2))**2)
                if (value<=cutoff):
                    H[u,v]=1
                else:
                    H[u,v]=0
    


        return H


    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        M,N=shape
        H=np.zeros((int(M),int(N)),np.float)
        for u in range(M):
            for v in range(N):
                value = math.sqrt((u-(M/2))**2 +(v-(N/2))**2)
                if (value<=cutoff):
                    H[u,v]=0
                else:
                    H[u,v]=1
    


        return H

        
       

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""
        M,N=shape
        H=np.zeros((int(M),int(N)),np.float)
        for u in range(M):
            for v in range(N):
                value = math.sqrt((u-(M/2))**2 +(v-(N/2))**2)
                n=2*order
                value1= (value/cutoff)**n
                H[u,v]=(1/(1+value1))
        return H

        
       

    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        M,N=shape
        H=np.zeros((int(M),int(N)),np.float)
        for u in range(M):
            for v in range(N):
             
                n=2*order
                if((u==M/2) and (v==N/2)):
                    u=M/2+1
                    v=N/2+1
                    value = math.sqrt((u-(M/2))**2 +(v-(N/2))**2)
                else:
                    value = math.sqrt((u-(M/2))**2 +(v-(N/2))**2)
                value1= (cutoff/value)**n
                H[int(u),int(v)]=(1/(1+value1))
        return H

        
       

    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""
        M,N=shape
        H=np.zeros((int(M),int(N)),np.float)
        for u in range(M):
            for v in range(N):
                value = math.sqrt((u-(M/2))**2 +(v-(N/2))**2)
                value1=(value**2/(2*cutoff*cutoff))
                H[u,v]=np.exp(-1*value1)
        return H

        
    

    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        M,N=shape
        H=np.zeros((int(M),int(N)),np.float)
        H2=self.get_gaussian_low_pass_filter(shape,cutoff)
        for u in range(M):
            for v in range(N):
                H[u,v]=1-H2[u,v]
                
        return H
        
       

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """
        g3=abs(image)
        
        B=np.amax(g3)
        A=np.amin(g3)
        
        value1=B-A
        M,N=image.shape
        g4=np.zeros((int(M),int(N)),np.float)
        
        for i in range(M):
            for j in range(N):
                constant=255/value1
                v1=constant*(g3[i,j]-A)
                g4[i,j]=int(v1)


        return g4


    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        take negative of the image to be able to view it (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8 
        """
        f1=np.fft.fft2(self.image)
        f2=np.fft.fftshift(f1)
        n=len(f1)
        f3=f2/n
        f3=abs(f3)
        f3=np.log(f3+1)
        dft_mag=10*f3
        im = Image.fromarray(dft_mag)
        im.show()
        
        shape=f1.shape
        
        
        if (self.filter=="butterworth_l"):
            H1=self.get_butterworth_low_pass_filter(shape,self.cutoff,self.order)
        elif (self.filter== "butterworth_h"):
            H1=self.get_butterworth_high_pass_filter(shape,self.cutoff,self.order)
        elif (self.filter=='ideal_l'):
            H1=self.get_ideal_low_pass_filter(shape,self.cutoff)
        elif (self.filter == 'ideal_h'):
            H1=self.get_ideal_high_pass_filter(shape,self.cutoff)
        elif (self.filter== 'gaussian_l'):
            H1=self.get_gaussian_low_pass_filter(shape,self.cutoff)
        elif (self.filter == 'gaussian_h'):
            H1=self.get_gaussian_high_pass_filter(shape,self.cutoff)
              
              

       
        
        #Spatial Convolution
        
        M,N=f2.shape
        
        g=np.zeros((int(M),int(N)),np.complex_)
        
        g=H1*f2
        
        n1=len(g)
        g5=g/n
        g5=abs(g5)
        g5=np.log(g5+1)
                
        filtered_dft_mag=10*g5
        
        im = Image.fromarray(filtered_dft_mag)
        im.show()
        
        g1=np.fft.ifftshift(g)
        
        g2=np.fft.ifft2(g1)
        
        output = self.post_process_image(g2)
        
        im = Image.fromarray(output)
        im.show()
        
        
        
        
        




        return [output, dft_mag, filtered_dft_mag]
