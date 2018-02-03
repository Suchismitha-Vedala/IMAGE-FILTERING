

1. DFT

The output for this is saved in ./output/dft_output folder.

a. Forward Fourier Transform.

Given an image I, we first calculate its shape.Let it be M,N
We then implement the following algorithm:(M=N)
for u in N:
	for v in N:
		each=0
		for i in N:
			for v in N:
				value=(ui+vj)*2*pi/N
				c1=math.cos(value)
				c2=math.sin(value)
				c3=I(i,j)*(c1-(1j*c2))
				each=each+c3
f[u,v]=each

This means at each step we calculate the summation real and imaginary parts of the whole image with the index of the required  image and  get the fourier transformed image.


This matrix is saved in FFT_{time}.txt				

b.Inverse Fourier Transform

The input to this is the output of a. 
Given an image I, we first calculate its shape.Let it be M,N
We then implement the following algorithm:(M=N)
for u in N:
	for v in N:
		each=0
		for i in N:
			for v in N:
				value=(ui+vj)*2*pi/N
				c1=math.cos(value)
				c2=math.sin(value)
				c3=I(i,j)*(c1+(1j*c2))
				each=each+c3
f[u,v]=each

This means at each step we calculate the summation real and imaginary parts of the whole image with the index of the required  image and  get the inverse fourier transformed image.


This matrix is saved in IFFT_{time}.txt	

c. Discrete Cosine Transform

As the name suggests, we find the cosine value of our DFT. 
We implement the following algorithm.
for u in N:
	for v in N:
		each=0
		for i in N:
			for v in N:
				value=(ui+vj)*2*pi/N
				c1=math.cos(value)
				c2=I(i,j)*c1
				each=each+c2
f[u,v]=each

This matrix is saved in Discrete_Cosine_{time}.txt	

d. Magnitude of DFT

The magnitude is nothing but the absolute of our inverse fourier transformed image.
We  calculate the magnitude for inverse fourier transformed image b.
for u in M:
	for v in N:
		f[u,v]= sqrt(real(u,v)**2 + imag(u,v)**2)
f is our required output

This matrix is saved in DFT_Magnitude_{time}.txt	


2. Frequency Filtering


To perform frequency filtering we do the following:

1. We compute the forward Fourier transform of given image. 
2. We then shift this image such that it is centralized at origin. Let this be f2.
This is our DFT.

Calculation of Magnitude of DFT:

We first normalize our f2 so that the values are quantifiable,
We then calculate the magnitude and perform the logarithmic expression on magnitude.
Since, log values are comparatively very small to fit in the range 0 to 255, we multiply it with a scalar to be able to visualize the result.
In this problem , let us take scalar as 10.

Calculation of Mask/Filter:

1. Ideal Low pass Filter:
2. Ideal High pass filter
3. Butterworth Low pass Filter
4. Butterworth High Pass Filter
5. Gaussian Low Pass Filter
6. Gaussian High Pass Filter 

Based on the mask name specified, filter is calculated.
The high pass filter is 1-low pass filter.

For ideal low pass filter:
mask =1 if D(u,v)<=cutoff
mask =0 otherwise.

D(u,v) = sqrt((u-M/2)**2+(v-N/2)**2) where M,N are share of the image.

Similarly other filters are calculated based on the functions.

Spatial Convolution:

This is done by multiplying the filter with f2 image. Let it be g. This is our filtered image.

Calculation of Magnitude of Filtered DFT:

We first normalize our g so that the values are quantifiable,
We then calculate the magnitude and perform the logarithmic expression on magnitude.
Since, log values are comparatively very small to fit in the range 0 to 255, we multiply it with a scalar to be able to visualize the result.
In this problem , let us take scalar as 10.


Post Processing of Image : 

Now we perform inverse shift on g, which gives g1.
g1 is inverse fourier transformed to get g2.

Full Contrast Stretch

We calculate the magnitude of g2.
We then calculate the max and min values. Let this be B,A.
We perform point wise linear stretch using the formula,

J(i,j)=255*(I(i,j)-A)/(B-A)

We round this to the nearest integer value
This returns our filtered image.

Interesting Results:

In the butterworth filter, as we increase the order, the mask size increases. 

