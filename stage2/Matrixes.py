from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage.filters import frangi
from scipy.ndimage import gaussian_filter


data1 = loadmat('/home/lyy/chenMLNovice/data/HW2data/Image_1064767_2_a0.mat')
data2 = loadmat('/home/lyy/chenMLNovice/data/HW2data/Image_1574121_2_a0.mat')

'''
print(data1.keys())

print(f'__header__:\n{data1['__header__']}')
print(f'__version__:\n{data1['__version__']}')
print(f'__globals__:\n{data1['__globals__']}')
print(f'Cells:\n{data1['Cells']}')
print(f'Image_out_voi:\n{data1['Image_out_voi']}')
print(f'R:\n{data1['R']}')
print(f'ResizeYN\n{data1['ResizeYN']}')
print(f'X_range\n{data1['X_range']}')
print(f'Y_range\n{data1['Y_range']}')
print(f'Z_range\n{data1['Z_range']}')
print(f'label\n{data1['label']}')
print(f'labelp\n{data1['labelp']}')
print(f'pos:\n{data1['pos']}')
'''

def Longest_d(points):
  # centered
  mean = np.mean(points, axis=0)
  centered_points = points - mean

  pca = PCA(n_components=3) # 3D
  pca.fit(centered_points)

  pc1 = pca.components_[0] # Eigenvector1
  pc2 = pca.components_[1] # Eigenvector2

  projections = np.dot(centered_points, pc1)

  longest_d = np.max(projections) - np.min(projections)

  return longest_d, pc1, pc2, centered_points, mean

def MIP_scatter(pc1, pc2, points):
  x_proj = np.dot(points, pc1)
  y_proj = np.dot(points, pc2)
  return x_proj, y_proj

def MIP_pixel(pc1, pc2, image):
  x, y, z = np.indices(image.shape)
  proj_x = x * pc1[0] + y * pc1[1] + z * pc1[2]
  proj_y = x * pc2[0] + y * pc2[1] + z * pc2[2]
    
  # 四捨五入並轉換為整數
  proj_x = np.round(proj_x).astype(int)
  proj_y = np.round(proj_y).astype(int)

  # let min == 0 
  proj_x -= proj_x.min()
  proj_y -= proj_y.min()

  max_x, max_y = proj_x.max() + 1, proj_y.max() + 1

  mip_image = np.zeros((max_x, max_y))
  np.maximum.at(mip_image, (proj_x.ravel(), proj_y.ravel()), image.ravel())

  return mip_image


'''Main'''
Image_out_voi = np.array(data1['Image_out_voi'])
R = np.array(data1['R']) # R is mask

masked_image = np.where(R == 1, Image_out_voi, 0)

# scatter
xs, ys, zs = np.nonzero(masked_image != 0) 
points = np.vstack((xs, ys, zs)).T # 加入行 並轉置矩陣

ld, pc1, pc2, centered_p, mean = Longest_d(points)
print(f"Longest diameter: {ld}")

x_proj_s, y_proj_s = MIP_scatter(pc1, pc2, centered_p)

# Real Image, not scatter
mip_image = MIP_pixel(pc1, pc2, Image_out_voi)
guass_image = gaussian_filter(mip_image, sigma=1.8) # Gaussian Filtering 
Frangi_image = frangi(guass_image)


'''Show Time'''
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# 展平 axes 陣列，方便逐一訪問
axes = axes.ravel()

# 1. Original 3D Point Cloud
ax1 = fig.add_subplot(221, projection='3d') 
ax1.scatter(xs, ys, zs, c='b', marker='o', s=10)
ax1.quiver(mean[0], mean[1], mean[2], pc1[0], pc1[1], pc1[2], color='r', length=50, label='PC1')
ax1.quiver(mean[0], mean[1], mean[2], pc2[0], pc2[1], pc2[2], color='g', length=50, label='PC2')
ax1.set_title('3D Point Cloud with PCA')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# 2. MIP Projection Along Principal Axis (Scatter)
axes[1].scatter(x_proj_s, y_proj_s, s=10)  
axes[1].set_title('MIP Projection (Scatter)')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

# 3. MIP Projection Along Principal Axis (Pixel)
axes[2].imshow(guass_image, cmap='gray')
axes[2].set_title('MIP Projection (Pixel)')
axes[2].axis('off')

# 4. Frangi Filter on MIP Projection
axes[3].imshow(Frangi_image, cmap='gray')
axes[3].set_title('Frangi Filtered MIP (Pixel)')
axes[3].axis('off')

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.tight_layout()
plt.show()


