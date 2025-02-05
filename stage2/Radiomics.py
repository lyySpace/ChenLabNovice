import numpy as np
import pickle
import matplotlib.pyplot as plt
from radiomics import featureextractor
import SimpleITK as sitk
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.decomposition import PCA

'''Gray-Level Co-Occurrence Matrix (GLCM) '''
def GLCM(img, distance = 1, orientation = 0): # 2D

    # 將影像的灰度值標準化為 0 到 n-1
    gray_levels = np.unique(img)
    gray_dict = {gray_levels[i]: i for i in range(len(gray_levels))}
    img = np.vectorize(lambda x: gray_dict[x])(img)

    # 初始化 GLCM 矩陣
    size = len(gray_levels)
    glcm = np.zeros((size, size), dtype=int)


    # 根據距離和方向來選擇鄰域
    if  orientation == 0:  # 水平
        dx, dy = distance, 0
    elif orientation == 45:  # 右上對角線
        dx, dy = distance, -distance
    elif orientation == 90:  # 垂直
        dx, dy = 0, distance
    elif orientation == 135:  # 左下對角線
        dx, dy = -distance, distance

    # 計算 GLCM
    for i in range(img.shape[0] - dx):
        for j in range(img.shape[1] - dy):
            x = img[i, j]
            y = img[i + dx, j + dy]
            # 將影像的灰度值直接用作索引
            glcm[x, y] += 1
            glcm[y, x] += 1  # GLCM 是對稱的

    glcm = glcm / np.sum(glcm) # [0 1]

    return glcm
   
def GLCM_features(img):
    glcm = GLCM(img, distance = 1, orientation = 0)
    # Contrast (Inertia)
    contrast = np.sum(np.square(np.indices(glcm.shape) - np.transpose(np.indices(glcm.shape))) * glcm)
    # Energy
    energy = np.sum(np.square(glcm))
    return contrast, energy

def GLCM_features_byPyradiomic(img, mask): # 3D
    # Convert NumPy array to SimpleITK image
    img = sitk.GetImageFromArray(img)  
    mask = sitk.GetImageFromArray(mask)

    # 創建 Pyradiomics 的特徵提取器
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # 提取特徵
    features = extractor.execute(img, mask)

    print("GLCM_features_byPyradiomic:")
    for featureName in features.keys():
        print(f"{featureName}: {features[featureName]}")

    return

def extract_radiomic_features(data):
    features = []
    labels = []
    for pickle in data:
        # 初始化特徵提取器
        extractor = featureextractor.RadiomicsFeatureExtractor()

        # 禁用所有特徵，僅啟用 GLCM
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('glcm')  # 啟用所有 GLCM 特徵

        # 讀取影像和遮罩
        img = sitk.GetImageFromArray(pickle['img'])
        mask = sitk.GetImageFromArray(pickle['mask'])

        # 提取特徵
        feature_vector = extractor.execute(img, mask)
            
        # 只提取 GLCM 特徵，避免 Pyradiomics 自動產生額外的 Metadata
        selected_features = {k: v for k, v in feature_vector.items() if 'glcm' in k.lower()}

        # 儲存特徵與標籤
        features.append([float(v) for v in selected_features.values()]) # np array To float
        labels.append(pickle['label'])
    
    return features, labels # list

'''Main'''

# 3
pickle_folder = '/home/lyy/chenMLNovice/data/LIDC_2_label_data/LIDC_label_test/'
pickle_path = [os.path.join(pickle_folder, f) for f in os.listdir(pickle_folder) if f.endswith('.pickle')]

# pickle_path = pickle_path[:100]  # 測試時僅使用前 50 個檔案

data = []
for p in pickle_path:
    with open(p, 'rb') as f:
        data.append(pickle.load(f))

features, labels = extract_radiomic_features(data)

'''Logistic Regression classification model '''
# 轉換成 DataFrame 以利處理
features_df = pd.DataFrame(features)
labels_df = pd.Series(labels)

# 資料標準化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)

# 如果特徵超過 2 維，則用 PCA 降維到 2 維
if features_scaled.shape[1] > 2:
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
else:
    features_2d = features_scaled  # 直接使用原始 2 維特徵

# 訓練 Logistic Regression # 直接使用所有數據來訓練模型 不切割
model = LogisticRegression()
model.fit(features_2d, labels_df)

# 繪製 Feature Space
plt.figure(figsize=(8,6))
sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels_df, palette='coolwarm', edgecolor='k')

# 繪製決策邊界
x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.xlabel("Feature 1 (PCA1)" if features_scaled.shape[1] > 2 else "Feature 1")
plt.ylabel("Feature 2 (PCA2)" if features_scaled.shape[1] > 2 else "Feature 2")
plt.title("Feature Space and Decision Boundary")
plt.legend(title="Label")
plt.show()


   
   
   
'''
# 2
# data1 is .pickle file
with open('/home/lyy/chenMLNovice/data/LIDC_2_label_data/LIDC_label_test/data_33.pickle', 'rb') as file: # 'rb' 二進位
    data1 = pickle.load(file) # 二進制資料反序列化
img1 = data1['img']
mask1 = data1['mask']
masked_img1 = np.where(mask1 == 1, img1, 0)

contrast, energy = GLCM_features(masked_img1[:,:,10])
print("Contrast:", contrast)
print("Energy:", energy)

GLCM_features_byPyradiomic(img1, mask1)
'''
