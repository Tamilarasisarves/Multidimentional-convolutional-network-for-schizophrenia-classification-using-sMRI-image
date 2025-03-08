#!/usr/bin/env python
# coding: utf-8

# ## Wavelet Decomposition and Feature extraction for 1DCNN model evaluation

# In[ ]:


import numpy as np
import pywt

def wavelet_transform(image, level=1, wavelet='db1'):
    transformed_slices = []
    for i in range(image.shape[0]):
        coeffs = pywt.wavedec2(image[i, :, :], wavelet, level=level)
        cA, (cH, cV, cD) = coeffs
        transformed_slices.append(cD)
    transformed_data = np.array(transformed_slices)
    return transformed_data
data_CD = wavelet_transform(data)
print(data_CD.shape, labels.shape) 


# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
def energy_calculation(x):
    squared_diff = tf.square(x[:, :, :, 1:] - x[:, :, :, :-1])
    energy = tf.reduce_sum(squared_diff, axis=(2, 3)) 
    return energy


model = keras.Sequential([
    layers.Input(shape=data_CD.shape[1:]), 
    layers.Lambda(energy_calculation)])

model.compile(optimizer='adam', loss='mean_squared_error')
data_CD_energy = model.predict(data_CD)
print(data_CD_energy.shape)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def create_model_CD():
    model = Sequential()
    model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(88, 1)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

model_CD_new = create_model_CD()

model_CD_new.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

model_CD_new.summary()


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

for train_index, val_index in kf.split(data_CD_energy):
    fold += 1
    print(f"Running Fold {fold}")

    X_train, X_val = data_CD_energy[train_index], data_CD_energy[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    tensorboard = TensorBoard(log_dir='logs')
    checkpoint = ModelCheckpoint("energy_CD.keras", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
    history_CD  = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=20, batch_size=16,
                        callbacks=[tensorboard, checkpoint])
    fold_score = model.evaluate(X_val, y_val)
    scores.append(fold_score)
    print(f"Fold {fold} score: {fold_score}")
print(f"Average score across all folds: {np.mean(scores)}")



# In[ ]:


from tensorflow.keras.models import load_model
model_CD_new.load_weights(r"D:\energy_CD.keras")


# ### Performance metrics, loss and accuracy graph

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc

all_fpr = []
all_tpr = []
roc_auc = []

history_acc = []
history_loss = []
history_val_acc = []
history_val_loss = []
all_accuracy = []


for fold in range(1, 6):  
    print(f"Evaluating Fold {fold}")
    fold_history = history_CD  
    history_acc.append(fold_history.history['accuracy'])
    history_val_acc.append(fold_history.history['val_accuracy']) 
    history_loss.append(fold_history.history['loss'])
    history_val_loss.append(fold_history.history['val_loss'])  
    
    y_val_pred = model_CD_new.predict(X_train)
    y_val_pred_bin = (y_val_pred > 0.5).astype(int)  
    
    acc = np.mean(y_val_pred_bin == y_train)  
    sensitivity = recall_score(y_train, y_val_pred_bin)  
    specificity = recall_score(y_train, y_val_pred_bin, pos_label=0)  
    f1 = f1_score(y_train, y_val_pred_bin)  
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")

   
    tn, fp, fn, tp = confusion_matrix(y_train, y_val_pred_bin).ravel()

    specificity_val = tn / (tn + fp) 
    sensitivity_val = tp / (tp + fn)  

    print(f"True Positive Rate (Sensitivity): {sensitivity_val:.4f}")
    print(f"True Negative Rate (Specificity): {specificity_val:.4f}")

    # Get ROC curve data
    fpr, tpr, thresholds = roc_curve(y_train, y_val_pred)
    auc_score = auc(fpr, tpr)
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    roc_auc.append(auc_score)

    all_accuracy.append(acc)  

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for fold_acc, fold_val_acc in zip(history_acc, history_val_acc):
    plt.plot(fold_acc, label=f'Training Accuracy (Fold {fold})')
    plt.plot(fold_val_acc, label=f'Validation Accuracy (Fold {fold})')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
for fold_loss, fold_val_loss in zip(history_loss, history_val_loss):
    plt.plot(fold_loss, label=f'Training Loss (Fold {fold})')
    plt.plot(fold_val_loss, label=f'Validation Loss (Fold {fold})')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# ### Visualizing Energy Distributions for Correct and Misclassified Predictions in Healthy and Schizophrenia Classes

# In[ ]:


pred_probs = model_CD_new.predict(data_CD_energy)
pred_labels = tf.argmax(pred_probs, axis=1)

cm = confusion_matrix(labels, pred_labels)
correct_indices = []
misclassified_indices = []
for i in range(len(labels)):
if pred_labels[i] == labels[i]:
correct_indices.append(i)
else:
misclassified_indices.append(i) 
print("Confusion Matrix:")
print(cm)
print(f"Correctly classified subjects: {correct_indices}")
print(f"Misclassified subjects: {misclassified_indices}")
correct_data = data_CD_energy[correct_indices] 
correct_energy = energy_model.predict(correct_data) 
print("Energy for correctly classified subjects:")
print(correct_energy)
misclassified_data = data_CD_energy[misclassified_indices] 
misclassified_energy = energy_model.predict(misclassified_data)
print("Energy for misclassified subjects:")
print(misclassified_energy)


# In[ ]:


import matplotlib.pyplot as plt
correct_healthy_energy = correct_energy[labels[correct_indices] == 0]
correct_schiz_energy = correct_energy[labels[correct_indices] == 1]
misclassified_healthy_energy = misclassified_energy[labels[misclassified_indices] == 0]
misclassified_schiz_energy = misclassified_energy[labels[misclassified_indices] == 1]
plt.figure(figsize=(14, 7))
data_to_plot = [correct_healthy_energy,correct_schiz_energy,misclassified_healthy_energy,misclassified_schiz_energy]
plt.boxplot(data_to_plot, labels=['Correct Healthy', 'Correct Schizophrenia',
'Misclassified Healthy', 'Misclassified Schizophrenia'])
plt.title('Energy Distribution by Classification and Class')
plt.ylabel('Energy')
plt.grid(True)
plt.show()

