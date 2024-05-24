# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# %%
red_image=cv2.imread('MainBefore.webp')
red=red_image[:,:,2]


plt.imshow(red)


# %%
green=red_image[:,:,1]
plt.imshow(green)

# %%
blue=red_image[:,:,0]
plt.imshow(blue)

# %%
#for RED MATRIX
M,N=red.shape
print(M,N)

# %%
lowquality=0.001*red

# %%
SR=np.zeros((M,N),dtype=np.float64)
print(SR)
threshold=np.sum(lowquality)/(M*N)

# %%
sigma=0.5  
losses=[]#to store the mean square difference

for _ in range(10):
    noisy=sigma*np.random.randn(M,N)+lowquality
    #modified image
    modified=np.where(noisy>threshold,255,0)
    
    mse=np.mean((modified-red)**2)#mean square error between the modified and the reference image
    losses.append(mse)
    SR+=modified
SR/=10



# %%
plt.subplot(2,2,1)
plt.title('reference red image')
plt.imshow(red,cmap='gray')

plt.subplot(2,2,2)
plt.title('lowquality')
plt.imshow(lowquality,cmap='gray')


plt.subplot(2,2,3)
plt.title('noisy')
plt.imshow(noisy,cmap='gray')

plt.subplot(2,2,4)
plt.title('SR')
plt.imshow(SR,cmap='gray')

# %%
#for green image
M,N=green.shape
print(M,N)

# %%
lowquality=0.001*green
SR2=np.zeros((M,N),dtype=np.float64)
print(SR2)
threshold=np.sum(lowquality)/(M*N)
sigma=0.5  
losses=[]#to store the mean square difference

for _ in range(10):
    noisy=sigma*np.random.randn(M,N)+lowquality
    #modified image
    modified=np.where(noisy>threshold,255,0)
    
    mse=np.mean((modified-green)**2)#mean square error between the modified and the reference image
    losses.append(mse)
    SR2+=modified
SR2/=10

# %%
plt.subplot(2,2,1)
plt.title('reference image')
plt.imshow(green,cmap='gray')

plt.subplot(2,2,2)
plt.title('lowquality')
plt.imshow(lowquality,cmap='gray')


plt.subplot(2,2,3)
plt.title('noisy')
plt.imshow(noisy,cmap='gray')

plt.subplot(2,2,4)
plt.title('SR')
plt.imshow(SR2,cmap='gray')

# %%
#for blue images

M,N=blue.shape
print(M,N)

lowquality=0.01*blue


SR3=np.zeros((M,N),dtype=np.float64)
print(SR3)
threshold=np.sum(lowquality)/(M*N)

sigma=0.5  
losses=[]#to store the mean square difference

for _ in range(10):
    noisy=sigma*np.random.randn(M,N)+lowquality
    #modified image
    modified=np.where(noisy>threshold,255,0)
    SR3+=modified
    mse=np.mean((modified-blue)**2)#mean square error between the modified and the reference image
    losses.append(mse)

SR3/=10

# %%
plt.subplot(2,2,1)
plt.title('reference image')
plt.imshow(blue,cmap='gray')

plt.subplot(2,2,2)
plt.title('lowquality')
plt.imshow(lowquality,cmap='gray')


plt.subplot(2,2,3)
plt.title('noisy')
plt.imshow(noisy,cmap='gray')

plt.subplot(2,2,4)
plt.title('SR')
plt.imshow(SR3,cmap='gray')

# %%
#from numpy import asarray
import numpy
numpydata=np.array(SR)
print(numpydata)

numpydata2=np.array(SR2)

numpydata3=np.array(SR3)

final_matrix=np.add(numpydata,numpydata2,numpydata3)
print(final_matrix)
img = numpy.zeros([720, 1280,3])
img[:,:,0] = final_matrix*64/255.0 #blue
img[:,:,1] = final_matrix*128/255.0 #green
img[:,:,2] = final_matrix*192/255.0 #red

cv2.imwrite('srgeneratedimage.jpg', img)
cv2.imshow("image", img)


