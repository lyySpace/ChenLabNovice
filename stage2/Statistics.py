import numpy as np
import pickle
import matplotlib.pyplot as plt
from radiomics import featureextractor
import SimpleITK as sitk
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import ttest_rel

def extract_radiomic_features(data):
    features = []
    labels = []
    for pickle in data:  # 3D
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('glcm')  

        img = sitk.GetImageFromArray(pickle['img'])
        mask = sitk.GetImageFromArray(pickle['mask'])

        feature_vector = extractor.execute(img, mask)

        # ensure only numbers
        selected_features = {k: v for k, v in feature_vector.items() if 'glcm' in k.lower()}

        features.append([float(v) for v in selected_features.values()])
        labels.append(pickle['label'])
    
    return np.array(features), np.array(labels)

def build_ANN_model(input_shape): # ANN Model
    '''
    Keras Sequential MLP 模型
        32 個 ReLU 神經元
        16 個 ReLU 神經元
        1 個 Sigmoid 輸出層 (二元分類)
    使用 Adam 優化器與 binary_crossentropy 損失函數
    '''
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_shape,)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def calculate_metrics(y_true, y_pred):
    """ 計算 Accuracy, Sensitivity (TPR), Specificity (TNR) """
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)  # TPR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp)  # TNR
    return acc, sens, spec

''' Load Data '''
# internal 
pickle_folder = '/home/lyy/chenMLNovice/data/LIDC_2_label_data/LIDC_label_test/'
pickle_path = [os.path.join(pickle_folder, f) for f in os.listdir(pickle_folder) if f.endswith('.pickle')]

data = []
for p in pickle_path:
    with open(p, 'rb') as f:
        data.append(pickle.load(f))

features, labels = extract_radiomic_features(data)

scaler = StandardScaler() # 標準化數據
features_scaled = scaler.fit_transform(features)

''' K-fold Cross-validation '''
kf = KFold(n_splits=5, shuffle=True, random_state=42) # setup

K_logistic_accuracy = []
K_ann_accuracy = []

K_logistic_FPR, K_logistic_TPR, K_logistic_AUC = [], [], []
K_ann_FPR, K_ann_TPR, K_ann_AUC = [], [], []

K_log_sensitivities, K_log_specificities = [], []
K_ann_sensitivities, K_ann_specificities = [], []

for train_idx, test_idx in kf.split(features_scaled):
    X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    # 1. Logistic Regression
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)

    y_pred_log = log_model.predict(X_test)
    K_logistic_accuracy.append(accuracy_score(y_test, y_pred_log))
    
    y_prob_log = log_model.predict_proba(X_test)[:, 1] 
    fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
    K_logistic_FPR.append(fpr_log)
    K_logistic_TPR.append(tpr_log)
    K_logistic_AUC.append(auc(fpr_log, tpr_log))

    log_acc, log_sens, log_spec = calculate_metrics(y_test, y_pred_log)
    K_log_sensitivities.append(log_sens)
    K_log_specificities.append(log_spec)
    
    # 2. ANN Model
    ann_model = build_ANN_model(X_train.shape[1])
    ann_model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    
    _, ann_acc = ann_model.evaluate(X_test, y_test, verbose=0)
    K_ann_accuracy.append(ann_acc)

    y_prob_ann = ann_model.predict(X_test).ravel()  
    fpr_ann, tpr_ann, _ = roc_curve(y_test, y_prob_ann)  
    K_ann_FPR.append(fpr_ann)
    K_ann_TPR.append(tpr_ann)
    K_ann_AUC.append(auc(fpr_ann, tpr_ann))

    y_pred_ann = (ann_model.predict(X_test) > 0.5).astype(int).ravel()
    ann_acc, ann_sens, ann_spec = calculate_metrics(y_test, y_pred_ann)
    K_ann_sensitivities.append(ann_sens)
    K_ann_specificities.append(ann_spec)


''' External Validation '''
# external data
external_pickle_folder = '/home/lyy/chenMLNovice/data/LIDC_2_label_data/LIDC_label_test/'
external_pickle_path = [os.path.join(external_pickle_folder, f) for f in os.listdir(external_pickle_folder) if f.endswith('.pickle')]

external_data = []
for p in external_pickle_path:
    with open(p, 'rb') as f:
        external_data.append(pickle.load(f))

external_features, external_labels = extract_radiomic_features(external_data)

scaler = StandardScaler() # 標準化數據
external_features_scaled = scaler.fit_transform(external_features)

# 1. Logistic Regression
log_model = LogisticRegression()
log_model.fit(features_scaled, labels)  
y_pred_log_external = log_model.predict(external_features_scaled)
Ex_logistic_accuracy  = accuracy_score(external_labels, y_pred_log_external)

y_prob_log_external = log_model.predict_proba(external_features_scaled)[:, 1]
Ex_logistic_FPR, Ex_logistic_TPR, _ = roc_curve(external_labels, y_pred_log_external)
Ex_logistic_AUC = auc(Ex_logistic_FPR, Ex_logistic_TPR)

Ex_logistic_acc, Ex_logistic_sens, Ex_logistic_spec = calculate_metrics(external_labels, y_pred_log_external)

# 2. ANN Model
ann_model = build_ANN_model(external_features_scaled.shape[1])
ann_model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
_, Ex_ann_accuracy = ann_model.evaluate(external_features_scaled, external_labels, verbose=0)

y_prob_ann_external = ann_model.predict(external_features_scaled).ravel()
Ex_ann_FPR, Ex_ann_TPR, _ = roc_curve(external_labels, y_prob_ann_external)
Ex_ann_AUC = auc(Ex_ann_FPR, Ex_ann_TPR)

Ex_ann_acc, Ex_ann_sens, Ex_ann_spec = calculate_metrics(external_labels, (y_prob_ann_external > 0.5).astype(int))

''' Statistical Comparison '''
# McNemar's Test
def mcnemar_test(y_true, y_pred1, y_pred2):
    """ 使用 McNemar 檢定比較兩個模型 """
    cm = confusion_matrix(y_pred1, y_pred2)
    result = mcnemar(cm, exact=True)
    return result.pvalue

p_value_acc_e = mcnemar_test(external_labels, y_pred_log_external, (y_prob_ann_external > 0.5).astype(int)) # Accuracy 

p_value_sens_e = mcnemar_test( # Sensitivity 
    external_labels[external_labels == 1],  
    y_pred_log_external[external_labels == 1],  
    (y_prob_ann_external[external_labels == 1] > 0.5).astype(int)
)

p_value_spec_e = mcnemar_test( # Specificity
    external_labels[external_labels == 0],  
    y_pred_log_external[external_labels == 0],  
    (y_prob_ann_external[external_labels == 0] > 0.5).astype(int)
)

# Paired T-test
t_stat, p_value_acc_i = ttest_rel(K_logistic_accuracy, K_ann_accuracy) # Accuracy 

t_stat, p_value_sens_i = ttest_rel(K_log_sensitivities, K_ann_sensitivities) # Sensitivity

t_stat, p_value_spec_i = ttest_rel(K_log_specificities, K_ann_specificities) # Specificity

t_stat, p_value_auc_i = ttest_rel(K_logistic_AUC, K_ann_AUC) # AUC


''' Result '''
# K-fold Validation (average)
print("-" * 50)
print("K-fold Cross-validation")
print(f'Logistic Regression Avg Accuracy: {np.mean(K_logistic_accuracy):.4f}')
print(f"Logistic Regression Avg Sensitivity: {np.mean(K_log_sensitivities):.4f}")
print(f"Logistic Regression Avg Specificity: {np.mean(K_log_specificities):.4f}")
print(f'ANN Avg Accuracy: {np.mean(K_ann_accuracy):.4f}') 
print(f"ANN Avg Sensitivity: {np.mean(K_ann_sensitivities):.4f}")
print(f"ANN Avg Specificity: {np.mean(K_ann_specificities):.4f}")
print("-" * 30)
print("Statistical Comparison")
print(f"Paired t-test - Accuracy: p = {p_value_acc_i:.4f}")
print(f"Paired t-test - Sensitivity: p = {p_value_sens_i:.4f}")
print(f"Paired t-test - Specificity: p = {p_value_spec_i:.4f}")
print(f"Paired t-test - AUC: p = {p_value_auc_i:.4f}")

# External Validation
print("-" * 50)
print("External Validation")
print(f'Logistic Regression Accuracy: {Ex_logistic_accuracy:.4f}')
print(f"Logistic Regression Sensitivity: {Ex_logistic_sens:.4f}")
print(f"Logistic Regression Specificity: {Ex_logistic_spec:.4f}")
print(f'ANN Accuracy: {Ex_ann_accuracy:.4f}')
print(f"ANN Sensitivity: {Ex_ann_sens:.4f}")
print(f"ANN Specificity: {Ex_ann_spec:.4f}")
print("-" * 30)
print("Statistical Comparison")
print(f"McNemar Test - Accuracy: p = {p_value_acc_e:.4f}")
print(f"McNemar Test - Sensitivity: p = {p_value_sens_e:.4f}")
print(f"McNemar Test - Specificity: p = {p_value_spec_e:.4f}")




'''ROC curve
Y: Sensitivity = TPR(True Positive Rate)
X: 1-Specificity = FPR (False Positive Rate)
'''
plt.figure(figsize=(10, 8))

# K-fold Validation (平均 ROC 曲線)
# Logistic Regression 插值法對齊
K_logistic_FPR_mean = np.linspace(0, 1, 100)  # 設定統一的 FPR 範圍
K_logistic_TPR_interp = np.zeros_like(K_logistic_FPR_mean)

for fpr, tpr in zip(K_logistic_FPR, K_logistic_TPR):
    K_logistic_TPR_interp += np.interp(K_logistic_FPR_mean, fpr, tpr) 

K_logistic_TPR_mean = K_logistic_TPR_interp / len(K_logistic_FPR)  # 取平均
K_logistic_AUC_mean = np.mean(K_logistic_AUC)

K_ann_FPR_mean = np.linspace(0, 1, 100)
K_ann_TPR_interp = np.zeros_like(K_ann_FPR_mean)

#  ANN 插值法對齊
for fpr, tpr in zip(K_ann_FPR, K_ann_TPR):
    K_ann_TPR_interp += np.interp(K_ann_FPR_mean, fpr, tpr)

K_ann_TPR_mean = K_ann_TPR_interp / len(K_ann_FPR)
K_ann_AUC_mean = np.mean(K_ann_AUC)


plt.plot(K_logistic_FPR_mean, K_logistic_TPR_mean, label=f'Logistic Regression (K-fold Avg AUC: {K_logistic_AUC_mean:.2f})', linewidth=2, linestyle='-', color='blue')
plt.plot(K_ann_FPR_mean, K_ann_TPR_mean, label=f'ANN (K-fold Avg AUC: {K_ann_AUC_mean:.2f})', linewidth=2, linestyle='-', color='green')

# External Validation
plt.plot(Ex_logistic_FPR, Ex_logistic_TPR, label=f'Logistic Regression (External AUC: {Ex_logistic_AUC:.2f})', color='red')
plt.plot(Ex_ann_FPR, Ex_ann_TPR, label=f'ANN (External AUC: {Ex_ann_AUC:.2f})', color='orange')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random guess)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity = FPR')
plt.ylabel('Sensitivity = TPR')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

