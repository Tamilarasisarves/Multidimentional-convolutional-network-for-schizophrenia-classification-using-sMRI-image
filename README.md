# Multidimentional-convolutional-network-for-schizophrenia-classification-using-sMRI-image
This repository contains Python scripts for classifying schizophrenia patients based on their structural MRI scans using a multidimensional convolutional network.
## How to Run:
First, run 2DCNN.py to prepare the data for the MDCNN model and train the 2D CNN model.
Next, run 1DCNN.py to perform feature extraction and train the 1D CNN model.
Then, run ROI_selection_energy_feature_extraction_analysis.py for ROI selection and feature extraction analysis.
Finally, use MVMDM.py to train the 3D CNN model and test its performance using the trained model.
