#!/usr/bin/env python
# coding: utf-8

# ## Extracting Specific Slices from NIfTI Files for Region of Interest (ROI) Analysis

# In[ ]:


import nibabel as nib
import numpy as np
import os
nifti_dir = (r"D:\data")

output_dir = (r"D:\roi")

roi_slices = {
    1: 44, 2: 56, 3: 58, 4: 62, 5: 68, 6: 70, 7: 74, 8: 80, 9: 86, 10: 88,
    11: 98, 12: 108, 13: 110, 14: 116, 15: 122, 16: 126, 17: 128, 18: 134,
    19: 138, 20: 140, 21: 152
}

for filename in os.listdir(nifti_dir):
    if filename.endswith('.nii') or filename.endswith('.nii.gz'):
        
        nifti_img = nib.load(os.path.join(nifti_dir, filename))
        
        nifti_data = nifti_img.get_fdata()

        combined_slices = []
        for roi, slice_num in roi_slices.items():
            slice_data = nifti_data[slice_num, :, :]
            combined_slices.append(slice_data)

     
        combined_slices = np.array(combined_slices)

        
        output_filename = os.path.join(output_dir, filename.replace('.nii', '_roi_slices.nii'))
        nib.save(nib.Nifti1Image(combined_slices, nifti_img.affine), output_filename)


# ### preparing data for energy feature extraction and plot analysis

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
data_CD = model.predict(data_CD)
print(data_CD.shape)


# ### Comparing Energy Distribution Across Brain Regions in Healthy vs. Schizophrenia Patients

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

healthy_indices = [i for i, label in enumerate(labels) if label == 0]
schizophrenia_indices = [i for i, label in enumerate(labels) if label == 1]
regions = [
    'Temporal Cortex_L', 'Parietal Cortex_L', 'Precentral_L', 'Insula_L', 
    'Lateral_Ventricle_L, Prefrontal Cortex_L', 'Putamen_L', 'Hippocampus_L', 
    'Amygdala_L', 'Thalamus_L, Caudate_L', 'Cingulate_Ant_L', 'Third_Ventricle, Fourth_Ventricle', 
    'Cingulate_Ant_R', 'Thalamus_R, Caudate_R', 'Amygdala_R', 'Hippocampus_R', 
    'Putamen_R', 'Lateral_Ventricle_R, Region Prefrontal Cortex_R', 'Insula_R', 
    'Precentral_R', 'Parietal Cortex_R', 'Temporal Cortex_R'
]


healthy_energy = energy_CD[healthy_indices]  
schizophrenia_energy = energy_CD[schizophrenia_indices] 

mean_healthy_energy = np.mean(healthy_energy, axis=0)
mean_schizophrenia_energy = np.mean(schizophrenia_energy, axis=0)

plt.figure(figsize=(12, 6))

plt.plot(regions, mean_healthy_energy, marker='o', linestyle='-', color='g', label='Healthy')

plt.plot(regions, mean_schizophrenia_energy, marker='o', linestyle='-', color='r', label='Schizophrenia')

plt.title('Energy vs. Region')
plt.xlabel('Region')
plt.ylabel('Energy Value')
plt.xticks(rotation=90)  
plt.grid(True)
plt.legend()  
plt.tight_layout() 
plt.show()


# In[ ]:


print("Healthy Energy Values per Slice:")
for i, value in enumerate(mean_healthy_energy):
    print(f"Slice {slices[i]}: {value:.2f}")

print("\nSchizophrenia Energy Values per Slice:")
for i, value in enumerate(mean_schizophrenia_energy):
    print(f"Slice {slices[i]}: {value:.2f}")


# ### Energy Distribution across Healthy and Schizophrenia Groups

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

healthy_energy = energy_CD[healthy_indices]
schizophrenia_energy = energy_CD[schizophrenia_indices] 

data = [healthy_energy.flatten(), schizophrenia_energy.flatten()]

plt.figure(figsize=(10, 6))

plt.boxplot(data, labels=['Healthy', 'Schizophrenia'], patch_artist=True, 
            boxprops=dict(facecolor='g', color='black'),
            flierprops=dict(markerfacecolor='r', marker='o', markersize=8))

plt.title('Energy Distribution across Healthy and Schizophrenia Groups')
plt.ylabel('Energy Value')
plt.grid(True)

plt.show()


# In[ ]:




