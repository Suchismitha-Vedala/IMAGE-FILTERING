# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import cv2
import sys
import numpy as np
from numpy.random import rand
from datetime import datetime
from numpy import zeros, newaxis
from matplotlib import pyplot as plt
import math,cmath

class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""
        a=matrix
        w,h = a.shape
        N=w
        c=((2*np.pi)/N)
        b=np.zeros((int(w),int(h)),np.complex)
        for i in range(int(w)):
            for j in range(int(h)):
                temp=0.0
                for u in range(int(w)):
                    for v in range(int(h)):
                       
                        n1=(u*i+v*j)
                        num=c*n1
                        #val1 = np.exp(-1j*num)
                        
                        c1=math.cos(num)
                        c2=math.sin(num)
                        #print c1,c2
                        val1 = (c1-(1j*c2))

                        c3_1=(a[i,j]*val1)
                        #print c3_1
                        temp+=c3_1

                        
                    
                
                b[u,v]=temp
        return b

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""
        b=matrix
        w,h = b.shape
        N=w
        c=np.zeros((int(w),int(h)),np.complex_)
        for i in range(int(w)):
            for j in range(int(h)):
                temp=0.0
                for u in range(int(w)):
                    for v in range(int(h)):
                        n2=(u*i+v*j)
                        num2=(2*math.pi*n2)/w
                        c1=math.cos(num2)
                        c2=math.sin(num2)
                        val2 = (c1+(1j*c2))
                        c3_2=(b[u,v]*val2)
                        temp+=c3_2
                
            
                
                c[i,j]=temp
                


        return c


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""
        
        a=matrix
        w,h = a.shape
        N=w
        cos=np.zeros((int(w),int(h)),np.complex_)
        for i in range(int(w)):
            for j in range(int(h)):
                temp=0.0
                for u in range(int(w)):
                    for v in range(int(h)):
                        n2=(u*i+v*j)
                        num2=(2*math.pi*n2)/w
                        c1=math.cos(num2)
                        c3_2=(a[u,v]*c1)
                        temp+=c3_2
                        
            
                
                cos[i,j]=temp
        




        return cos


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""
        
        b=matrix
        w,h = b.shape
       
   
        f = np.zeros((w,h))
        for u in range(w):
            for v in range(h):
                f[u,v]=np.sqrt((np.square(b[u,v].real)) + (np.square(b[u,v].imag)))
                   
        return f