import numpy as np
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

def Histogram(img):
  hist = np.zeros(shape=(img.max()+1)) # vector 

  for rows in img: # a row, nparray
    for element in rows:
      hist[element] += 1 

  hist_nrom = hist/hist.sum() # Normalized into [0,1]
  return hist_nrom # vector 

'''Cumulative Distribution Function(CDF)'''
def CDF(img):
  cdf = np.zeros(shape=(img.max()+1)) # vector 

  hist = Histogram(img)

  for idx, element in enumerate(hist):
    if idx == 0:
      cdf[idx] += element
    else:
      cdf[idx] += (element + cdf[idx-1]) # 累加

  return cdf # vector 

'''Histogram Equalization'''
def HistogramEqual(img_input,L):
  img_out = np.zeros(shape=img_input.shape) 
  cdf = CDF(img_input) # vector 

  for rows, element_in_col in enumerate(img_input):
    for cols, element in enumerate(element_in_col):
      img_out[rows, cols] = L*cdf[element] # The transform function

  return img_out

'''Morphological Image Processing'''
def erosion(img, kernal):
  pad_h, pad_w = kernal.shape[0]//2, kernal.shape[1]//2
  padded_img = np.zeros((img.shape[0]+2*pad_h, img.shape[1]+2*pad_w))
  padded_img[pad_h:-pad_h, pad_w:-pad_w] = img

  eroded_img = np.zeros(shape=img.shape)

  for i in range(pad_h, padded_img.shape[0]-pad_h):
    for j in range(pad_w, padded_img.shape[1]-pad_w):
      region = padded_img[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]
      eroded_img[i-pad_h, j-pad_w] = np.min(region[kernal==1])

  return eroded_img

def dilation(img, kernal):
  pad_h, pad_w = kernal.shape[0]//2, kernal.shape[1]//2
  padded_img = np.zeros((img.shape[0]+2*pad_h, img.shape[1]+2*pad_w))
  padded_img[pad_h:-pad_h, pad_w:-pad_w] = img

  dilated_img = np.zeros(shape=img.shape)

  for i in range(pad_h, padded_img.shape[0]-pad_h):
      for j in range(pad_w, padded_img.shape[1]-pad_w):
          region = padded_img[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]
          dilated_img[i-pad_h, j-pad_w] = np.max(region[kernal==1])

  return dilated_img

def opening(img, kernal):
  opened_img = dilation(erosion(img, kernal), kernal)
  return opened_img

def closing(img, kernal):
  closed_img = erosion(dilation(img, kernal), kernal)
  return closed_img

def gradient(img, kernal):
  gradient_img = dilation(img, kernal) - erosion(img, kernal)
  return gradient_img

'''Geometric Transformations'''
def affine_transform(img, A):
  affine_img = np.zeros(shape=img.shape)

  inverse_A = np.linalg.inv(A)

  for i in range(img.shape[0]): # row
    for j in range(img.shape[1]): # columm
      point = np.array([i, j, 1])
      new_point = np.matmul(inverse_A, point)

      r = min(max(new_point[0], 0), img.shape[0]-1)
      c = min(max(new_point[1], 0), img.shape[1]-1)

      # neighbors
      r_floor, c_floor = np.floor(r).astype(int), np.floor(c).astype(int)
      r_ceil, c_ceil = np.ceil(r).astype(int), np.ceil(c).astype(int)

      r_floor = max(min(r_floor, img.shape[0] - 1), 0)
      c_floor = max(min(c_floor, img.shape[1] - 1), 0)
      r_ceil = max(min(r_ceil, img.shape[0] - 1), 0)
      c_ceil = max(min(c_ceil, img.shape[1] - 1), 0)

      f_00 = img[r_floor, c_floor]
      f_01 = img[r_floor, c_ceil]
      f_10 = img[r_ceil, c_floor]
      f_11 = img[r_ceil, c_ceil]

      if new_point[0] > 0 and new_point[0] < img.shape[0] and new_point[1] > 0 and new_point[1] < img.shape[1]:
        if r_ceil != r_floor and c_ceil != c_floor:
          f_r1 = (f_10 * (c_ceil - c) + f_11 * (c- c_floor)) / (c_ceil - c_floor)
          f_r2 = (f_00 * (c_ceil - c) + f_01 * (c- c_floor)) / (c_ceil - c_floor)
          affine_img[i, j] = (f_r1 * (r - r_floor) + f_r2 * (r_ceil - r)) / (r_ceil - r_floor)
        elif r_ceil != r_floor and c_ceil == c_floor:
          affine_img[i, j] = (f_01 * (r_ceil - r) + f_11 * (r - r_floor)) / (r_ceil - r_floor)
        elif r_ceil == r_floor and c_ceil != c_floor:
          affine_img[i, j] = (f_11 * (c_ceil - c) + f_10 * (c - c_floor)) / (c_ceil - c_floor)
        else:
          affine_img[i, j] = f_00
      else:
        affine_img[i, j] = 0

  return affine_img

# HW1,2
image_path = '/home/lyy/chenMLNovice/data/pout.jpg'
img = cv.imread(image_path)
kernel = np.ones((3,3),np.uint8)
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # BRG to grey
img_ = gray_img/255

img_opening = opening(img_, kernel)
img_closing = closing(img_, kernel)
img_gradient = gradient(img_, kernel)

titles = ['Original Image', 'Opening', 'Closing', 'Morphological Gradient']
images = [img, img_opening, img_closing, img_gradient]

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# HW3,4
degree = (-30*math.pi)/180
affine_matrix = np.array([[math.cos(degree),-math.sin(degree),0],[math.sin(degree),math.cos(degree),0],[0,0,1]])
print(affine_matrix)
img_affine = affine_transform(gray_img, affine_matrix)
img_affine = img_affine.astype(np.uint8) #np.uint8, ensuring it has the correct depth expected by the SIFT

'''The SIFT algorithm '''
# 初始化 SIFT 檢測器
sift = cv.SIFT_create()

# 檢測特徵點與描述符
kp1, des1 = sift.detectAndCompute(img_affine, None)
kp2, des2 = sift.detectAndCompute(img, None)

# 使用 FLANN 匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 篩選匹配點
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 提取匹配點座標
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

'''A_New'''
A_new, inliers = cv.estimateAffinePartial2D(src_pts, dst_pts, method=cv.RANSAC)
A_new = A_new.astype(np.float32)  # Convert the transformation matrix to float32
A_new[:,2] = 0
A_new = np.vstack([A_new, [0, 0, 1]]) # A_new with a row of [0, 0, 1] to make it a 3x3 square matrix
print(A_new)

new_img_affine = affine_transform(img, A_new)

fig, axes = plt.subplots(1,3,figsize=(15,8))
axes[0].imshow(img,cmap='gray',vmin=0,vmax=255)
axes[0].axis('off')
axes[0].set_title('Original image', fontweight='bold')
axes[1].imshow(img_affine,cmap='gray',vmin=0,vmax=255)
axes[1].axis('off')
axes[1].set_title('Affine transform image', fontweight='bold')
axes[2].imshow(new_img_affine,cmap='gray',vmin=0,vmax=255)
axes[2].axis('off')
axes[2].set_title('New Affine transform image', fontweight='bold')
plt.show()