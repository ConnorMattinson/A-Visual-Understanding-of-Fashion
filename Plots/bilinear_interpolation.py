import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

# Figure 34
image = np.zeros((10,10,3))
image[0,0,0] = 1
image[-1,-1,1] = 1
image[-1,0,2] = 1
image[0,-1,1] = 1
image[0,-1,2] = 1

quantised = image
quantised[:5,:5] = np.array([[[1,0,0]]*5]*5)
quantised[:5,5:] = np.array([[[0,1,1]]*5]*5)
quantised[5:,5:] = np.array([[[0,1,0]]*5]*5)
quantised[5:,:5] = np.array([[[0,0,1]]*5]*5)
img_up_red = np.zeros((10,10,3))
img_up_cyan = np.zeros((10,10,3))
img_up_blue = np.zeros((10,10,3))
img_up_green = np.zeros((10,10,3))
for i in range(10):
    for j in range(10):
        img_up_red[i,j] = ((10-i)/10)*image[0,0] 
        
for i in range(10):
    for j in range(10):
        img_up_cyan[i,j] = ((10-i)/10)*image[0,9] 
        
for i in range(10):
    for j in range(10):
        img_up_blue[i,j] = (i/10)*image[9,0] 
       
for i in range(10):
    for j in range(10):
        img_up_green[i,j] = (i/10)*image[9,9]

img_side_red = np.zeros((10,10,3))
img_side_cyan = np.zeros((10,10,3))
img_side_blue = np.zeros((10,10,3))
img_side_green = np.zeros((10,10,3))
for i in range(10):
    for j in range(10):
        img_side_red[i,j] = ((10-j)/10)*img_up_red[i,j]

for i in range(10):
    for j in range(10):
        img_side_blue[i,j] = ((10-j)/10)*img_up_blue[i,j]        

for i in range(10):
    for j in range(10):
        img_side_cyan[i,j] = (j/10)*img_up_cyan[i,j]
    
for i in range(10):
    for j in range(10):
        img_side_green[i,j] = (j/10)*img_up_green[i,j]

plt.axis('off')
plt.imshow(image)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.imshow(img_up_red)
ax1.axis('off')
ax2.imshow(img_up_cyan)
ax2.axis('off')
ax3.imshow(img_up_blue)
ax3.axis('off')
ax4.imshow(img_up_green)
ax4.axis('off')

fig2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2)
ax21.imshow(img_side_red)
ax21.axis('off')
ax22.imshow(img_side_cyan)
ax22.axis('off')
ax23.imshow(img_side_blue)
ax23.axis('off')
ax24.imshow(img_side_green)
ax24.axis('off')

plt.axis('off')
plt.imshow(img_side_red + img_side_cyan + img_side_blue + img_side_green)

plt.axis('off')
plt.imshow(quantised)
