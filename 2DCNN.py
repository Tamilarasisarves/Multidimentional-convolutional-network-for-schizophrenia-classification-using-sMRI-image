#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[6]:


import nibabel as nib
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import tensorflow.keras
from keras import ops
import keras
from keras import layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Activation
from tensorflow.keras.layers import BatchNormalization
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D,MaxPooling3D,Flatten,Dense,Activation,Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras  import regularizers
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,  ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard


# ### MNI Space Registration of Structural MRI Data

# In[ ]:


import os
import nibabel as nib
from nilearn import image, plotting
from nilearn import datasets

mni_template = datasets.load_mni152_template()

directory = "/content/drive/MyDrive/smri/"
save_directory = "/content/sample_data/registered/"
os.makedirs(save_directory, exist_ok=True)
files = [f for f in os.listdir(directory) if f.endswith('.nii') or f.endswith('.nii.gz')]
for file in files:
    sMRI_path = os.path.join(directory, file)
    sMRI_img = nib.load(sMRI_path)
    registered_img = image.resample_to_img(sMRI_img, mni_template)
    save_path = os.path.join(save_directory,file)
    registered_img.to_filename(save_path)
    plotting.plot_img(registered_img, title=f"Registered: {file}")

print("Registration complete for all files.")


# ### Processing NIfTI Files: Extracting Slices

# In[ ]:


import os
import nibabel as nib
import numpy as np
nifti_dir=("/content/sample_data/registered")
for filename in os.listdir(nifti_dir):
    if filename.endswith('.nii') or filename.endswith('.nii.gz'):
        nifti_img = nib.load(os.path.join(nifti_dir, filename))
        nifti_data = nifti_img.get_fdata()
        selected_slices = nifti_data[44:132, :, :]
        output_filename = os.path.join(output_dir, filename.replace('.nii', '_slices_88.nii'))
        nib.save(nib.Nifti1Image(selected_slices, nifti_img.affine), output_filename)


# ### Accessing Data Directory

# In[ ]:


import os
import numpy as np
healthy_directory = (r"D:\smri-88\healthy/")
healthy_files = os.listdir(healthy_directory)


# In[ ]:


import os
import numpy as np
schiz_directory =(r"D:\smri-88\schizophrenia/")
schiz_files = os.listdir(schiz_directory)


# ### Reading and Storing MRI Images

# In[ ]:


healthy_images=[]
import nibabel as nib
for file1 in healthy_files:
    if file1.endswith('.nii'):
        healthy_image= nib.load(healthy_directory+file1)
        healthy_image= healthy_image.get_fdata()
        healthy_images.append(healthy_image)


# In[ ]:


schiz_images=[]
import nibabel as nib
for file2 in schiz_files:
    if file2.endswith('.nii'):
        schiz_image= nib.load(schiz_directory+file2)
        schiz_image= schiz_image.get_fdata()
        schiz_images.append(schiz_image)


# ### Data Preparation

# In[ ]:


from sklearn.model_selection import train_test_split
import numpy as np
import pywt
import matplotlib.pyplot as plt

data = np.concatenate((healthy_images, schiz_images), axis=0)
labels = np.concatenate((np.zeros(len(healthy_images)), np.ones(len(schiz_images))), axis=0)
print(data.shape, labels.shape)


# ### Normalization

# In[ ]:


mean = data.mean()
std = data.std()
data = (data - mean) / std


# ### DWT and 2DCNN model

# In[ ]:


import numpy as np
import pywt

def wavelet_transform(image, level=1, wavelet='db1'):
    transformed_slices = []
    for i in range(image.shape[0]):
        coeffs = pywt.wavedec2(image[i, :, :], wavelet, level=level)
        cA, (cH, cV, cD) = coeffs
        transformed_slices.append(cH)
    transformed_data = np.array(transformed_slices)
    return transformed_data

data_CH = wavelet_transform(data)
print(data_CH.shape, labels.shape) 


# In[ ]:


data_CH = np.mean(data_CH , axis=(1))
print(data_CH.shape, labels.shape)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout

def create_model_CH():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

model_CH_new = create_model_CH()

model_CH_new.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

model_CH_new.summary()


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

for train_index, val_index in kf.split(data_CH):
    fold += 1
    print(f"Running Fold {fold}")

    X_train, X_val = data_CH[train_index], data_CH[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    tensorboard = TensorBoard(log_dir='logs')
    checkpoint = ModelCheckpoint("CH.keras", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
    history_CH  = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=20, batch_size=16,
                        callbacks=[tensorboard, checkpoint])
    fold_score = model.evaluate(X_val, y_val)
    scores.append(fold_score)
    print(f"Fold {fold} score: {fold_score}")
print(f"Average score across all folds: {np.mean(scores)}")


# ### performance metrics

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
model_CH.load_weights("CH.keras")

results_val = model_CH.evaluate(val_data, val_labels)
print("Validation Accuracy - CH:", results_val[1])

predictions_val = model_CH.predict(val_data)
predictions_binary = (predictions_val > 0.5).astype('int')

cm_val = confusion_matrix(val_labels, predictions_binary)
accuracy_val = accuracy_score(val_labels, predictions_binary)
precision_val = precision_score(val_labels, predictions_binary)
recall_val = recall_score(val_labels, predictions_binary)
f1_val = f1_score(val_labels, predictions_binary)
TN_val, FP_val, FN_val, TP_val = cm_val.ravel()
sensitivity_val = TP_val / (TP_val + FN_val)
specificity_val = TN_val / (TN_val + FP_val)
print("Confusion Matrix (Validation):")
print(cm_val)
print("Validation Accuracy:", accuracy_val)
print("Validation Precision:", precision_val)
print("Validation Recall (Sensitivity):", recall_val)
print("Validation Specificity:", specificity_val)
print("Validation F1 Score:", f1_val)


# ### Evaluating Model Predictions

# In[ ]:


from tensorflow.keras.models import load_model
model_CH_new.load_weights(r"CH.keras")
predictions_binary = predictions_binary.flatten()

correctly_classified = np.where(labels == predictions_binary)[0]
misclassified = np.where(labels != predictions_binary)[0]

print(f"Number of Correctly Classified Samples: {len(correctly_classified)}")
print(f"Number of Misclassified Samples: {len(misclassified)}")


# In[ ]:


import numpy as np

model_output = model_CH_new.predict(data_CH)  

confidence_scores = model_output.flatten()  

predictions_binary = (confidence_scores >= 0.5).astype(int)  

labels = labels.flatten()  

correctly_classified = np.where(labels == predictions_binary)[0]
misclassified = np.where(labels != predictions_binary)[0]

correct_confidence = confidence_scores[correctly_classified]
misclassified_confidence = confidence_scores[misclassified]

print(f"Number of Correctly Classified Samples: {len(correctly_classified)}")
print(f"Number of Misclassified Samples: {len(misclassified)}")
print(f"Average Confidence (Correct): {np.mean(correct_confidence):.4f}")
print(f"Average Confidence (Misclassified): {np.mean(misclassified_confidence):.4f}")
print(f"Confidence Scores of Correctly Classified Samples: {correct_confidence}")
print(f"Confidence Scores of Misclassified Samples: {misclassified_confidence}")

confidence_gap = np.mean(correct_confidence) - np.mean(misclassified_confidence)
print(f"Confidence Gap (Correct vs. Misclassified): {confidence_gap:.4f}")


# ### Grad-CAM Vizualization

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model = load_model("CH.keras")
def compute_gradcam(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        if pred_index is None:
            pred_index = tf.argmax(predictions, axis=-1).numpy()[0]

        class_channel = predictions[0, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, img):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.jet
    jet_heatmap = jet(heatmap)
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    if jet_heatmap.shape[-1] == 4:
        jet_heatmap = jet_heatmap[..., :3]

    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)

    superimposed_img = jet_heatmap * 0.4 + img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

predictions = model.predict(data_CH)
predicted_classes = np.argmax(predictions, axis=-1)

schizophrenia_indices = np.where(predicted_classes == 1)[0]

num_images = len(schizophrenia_indices[:20])
fig, axs = plt.subplots(num_images, 2, figsize=(15, num_images * 5))

last_conv_layer_name = 'conv_2d' 

for i, idx in enumerate(schizophrenia_indices):
    img = data_CH[idx]
    img_array = np.expand_dims(img, axis=0)
    if img.ndim == 2:  
        img = np.stack((img,) * 3, axis=-1)  
    heatmap = compute_gradcam(model, img_array, last_conv_layer_name)
    superimposed_img = overlay_heatmap(heatmap, img)
    axs[i, 0].imshow(image.array_to_img(img))  
    axs[i, 0].set_title(f'Original Image {idx}')
    axs[i, 0].axis('off')

    axs[i, 1].imshow(superimposed_img)
    axs[i, 1].set_title(f'Grad-CAM for Image {idx}')
    axs[i, 1].axis('off')

plt.tight_layout()
plt.show()

