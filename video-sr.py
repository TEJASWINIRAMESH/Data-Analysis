# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

# %%
import os
from os import listdir

# %%
cam = cv2.VideoCapture("C:\\Users\\student.ITLAB\\Downloads\\3588882-hd_1920_1080_18fps.mp4") 
try: 	 
	if not os.path.exists('data'): 
		os.makedirs('data')  
except OSError: 
	print ('Error: Creating directory of data') 
 
currentframe = 0
while(True): 	
	ret,frame = cam.read() 
	if ret: 	
		name = './data/frame' + str(currentframe) + '.jpg'
		print ('Creating...' + name) 
		cv2.imwrite(name, frame) 
		currentframe += 1
	else: 
		break

cam.release() 
cv2.destroyAllWindows() 


# %%
refer=cv2.imread('C:\\Users\\student.ITLAB\\Desktop\\112\data\\frame0.jpg',cv2.IMREAD_GRAYSCALE)


# %%
sigma=0.5
folder_dir = "C:\\Users\\student.ITLAB\\Desktop\\112\\data"
for image in os.listdir(folder_dir):
    path="C:\\Users\\student.ITLAB\\Desktop\\112\\data\\"+image
    dest="C:\\Users\\student.ITLAB\\Desktop\\112\\sr_data\\"+image
    refer=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    M,N=refer.shape
    low_quality=refer*0.01
    threshold=np.sum(low_quality)/(M*N)
    sr=np.zeros((M,N),dtype=np.float64)
    for i in range(5):
        noisy=(np.random.randn(M,N)*sigma)+low_quality
        modified=np.where(noisy>threshold,255,0)
        sr+=modified
    
    cv2.imwrite(dest, sr)



# %%
def create_video_from_images(folder):
    video_filename = 'created_video.mp4'
    valid_images = [i for i in os.listdir(folder) if i.endswith((".jpg", ".jpeg", ".png"))]

    first_image = cv2.imread(os.path.join(folder, valid_images[0]))
    h, w, _ = first_image.shape

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(video_filename, codec, 30, (w, h))

    for img in valid_images:
        loaded_img = cv2.imread(os.path.join(folder, img))
        for _ in range(20):
            vid_writer.write(loaded_img)

    vid_writer.release()

# %%
create_video_from_images("C:\\Users\\student.ITLAB\\Desktop\\112\\sr_data")

# %%



