#!/usr/bin/env python
# coding: utf-8

# ## Wavelet Decomposition and 3DCNN model 

# In[ ]:


import numpy as np
import pywt

def wavelet_transform(image, level=1, wavelet='db1'):
    transformed_slices = []
    for i in range(image.shape[0]):
        coeffs = pywt.wavedec2(image[i, :, :], wavelet, level=level)
        cA, (cH, cV, cD) = coeffs
        transformed_slices.append(cV)
    transformed_data = np.array(transformed_slices)
    return transformed_data

data_CV = wavelet_transform(data)
print(data_CV.shape, labels.shape) 


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model_CV():
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(2, 3, 3), activation='relu', input_shape=( 88, 128, 128, 1)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Conv3D(64, kernel_size=(2, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Conv3D(64, kernel_size=(2, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Conv3D(64, kernel_size=(2, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

model_CV_new = create_model_CV()

model_CV_new.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

model_CV_new.summary()


# ### 5-Fold cross validation

# In[ ]:


import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint


kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
fold = 0

for train_index, val_index in kf.split(data_CV):
    fold += 1
    print(f"Running Fold {fold}")

    X_train, X_val = data_CV[train_index], data_CV[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    tensorboard = TensorBoard(log_dir='logs')
    checkpoint = ModelCheckpoint("CV.keras", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
    history_CV  = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=20, batch_size=16,
                        callbacks=[tensorboard, checkpoint])
    fold_score = model.evaluate(X_val, y_val)
    scores.append(fold_score)
    print(f"Fold {fold} score: {fold_score}")
print(f"Average score across all folds: {np.mean(scores)}")


# In[ ]:


from tensorflow.keras.models import load_model
model_CV_new.load_weights("CV.keras")


# ### ROC curve

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
predictions_CV = model_CV_new.predict(X_train)
fpr, tpr, thresholds = roc_curve(y_train, predictions_CD)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random classifier)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# ### Extracting Spatial Features from each 3D CNN Layer

# In[ ]:


from tensorflow.keras.models import Model
layer_name = 'conv3d_4'  
spatial_feature_extractor_model = Model(inputs=model_CV_new.input, outputs=model_CV_new.get_layer(layer_name).output)
sample_image = data_CV[0]  
spatial_features = spatial_feature_extractor_model.predict(sample_image[np.newaxis]) 
spatial_features.shape


# In[ ]:


import matplotlib.pyplot as plt
spatial_features_for_visualization = spatial_features_1[0, 44, :, :, :]  
fig, axes = plt.subplots(8, 8, figsize=(16, 16)) 
for i in range(8):
    for j in range(8):
        ax = axes[i, j]
        ax.imshow(spatial_features_for_visualization[:, :, i * 2 + j], cmap='Oranges') 
        ax.axis('off')

plt.show()


# ### MVMDM model prediction and evaluation

# In[ ]:


from tensorflow.keras.models import load_model
model_CD_new=load_model("D:\energy_CD.keras")
model_CH_new=load_model("D:\CH.keras")
model_CV_new=load_model("D:\CV.keras")


# In[ ]:


import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pywt

folder_path = "D:\smri_data"

for filename in os.listdir(folder_path):
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        file_path = os.path.join(folder_path, filename)
        
        brain = nib.load(file_path)
        brain_data = brain.get_fdata()
        coeffs_photo = pywt.wavedec2(brain_data, 'db1', level=1)
        cA_photo, (cH_photo, cV_photo, cD_photo) = coeffs_photo
        
        def energy_calculation(x):
            squared_diff = tf.square(x[:, 1:, :] - x[:, :-1, :]) 
            energy = tf.reduce_sum(squared_diff, axis=(1,2)) 
            return energy
       
        energy_model = keras.Sequential([
            layers.Input(shape=cD_photo.shape[1:]),
            layers.Lambda(energy_calculation)])
        
        cD_photo_energy = energy_model.predict(cD_photo)
         
        cH_photo_mean = np.mean(cH_photo, axis=0)
        
        cD_photo_input = cD_photo_energy[np.newaxis, ..., np.newaxis]
        cH_photo_input = cH_photo_mean[np.newaxis, ..., np.newaxis]
        cV_photo_input = cV_photo[np.newaxis, ..., np.newaxis]

        prediction_CD_new = model_CD_new.predict(cD_photo_input)
        prediction_CH_new = model_CH_new.predict(cH_photo_input)
        prediction_CV_new = model_CV_new.predict(cV_photo_input)
       
        combined_prediction = (prediction_CD_new+ prediction_CH_new +  prediction_CV_new) / 3
        
        final_prediction = combined_prediction > 0.5

        if final_prediction:
            print(f"The brain data in {filename} is predicted to be related to schizophrenia.")
        else:
            print(f"The brain data in {filename} is predicted to be healthy.")



# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr_CH, tpr_CH, _ = roc_curve(labels, model_CH_new.predict(cH_photo_input))
roc_auc_CH = auc(fpr_CH, tpr_CH)

fpr_CD, tpr_CD, _ = roc_curve(labels, model_CD_new.predict(cD_photo_input))
roc_auc_CD = auc(fpr_CD, tpr_CD)

fpr_CV, tpr_CV, _ = roc_curve(labels, model_CV_new.predict(cV_photo_input))
roc_auc_CV = auc(fpr_CV, tpr_CV)
plt.figure(figsize=(8, 6))


plt.plot(fpr_CH, tpr_CH, color='darkorange', lw=2, label=f'Model CH (AUC = {roc_auc_CH:.2f})')
plt.plot(fpr_CD, tpr_CD, color='green', lw=2, label=f'Model CD (AUC = {roc_auc_CD:.2f})')
plt.plot(fpr_CV, tpr_CV, color='blue', lw=2, label=f'Model CV (AUC = {roc_auc_CV:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Comparison of ROC Curves for Different Models')
plt.legend(loc='lower right')
plt.show()

