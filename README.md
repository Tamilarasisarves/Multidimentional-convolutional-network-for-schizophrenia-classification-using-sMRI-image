# Multidimentional-convolutional-network-for-schizophrenia-classification-using-sMRI-image
This repository contains Python scripts for classifying schizophrenia patients based on their structural MRI scans using a multidimensional convolutional network.
This repository explores the use of multidimensional Convolutional Neural Network (CNN) architectures, specifically 1D-CNN, 2D-CNN, and 3D-CNN, for classifying schizophrenia based on structural MRI (sMRI) data. The study uses Discrete Wavelet Transform (DWT) subbands to extract energy features, which highlight various frequency elements in the data related to schizophrenia.

1D-CNN: Focuses on energy features extracted from the CD subband, which emphasizes diagonal high-frequency components, known to be associated with schizophrenia.
2D-CNN: Uses the CH subband, which allows for feature extraction from horizontal high-frequency coefficients of the sMRI data.
3D-CNN: Utilizes the CV subband, enabling volumetric feature extraction from vertical high-frequency coefficients. The proposed method uses ensembling strategy (max voting) from the multidimensional CNN classification outputs to improve the classification performance. 
## Data Availability 

The datasets are from the following open repositories:

- **UCLA Consortium for Neuropsychiatric Phenomics LA5c Study database**: [https://openfmri.org/dataset/ds000030/](https://openfmri.org/dataset/ds000030/)
- **OpenfMRI dataset for schizophrenia**: [https://openfmri.org/dataset/ds000115/](https://openfmri.org/dataset/ds000115/)
- **COBRE dataset**: [https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html](https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html)

## How to Run:

1. **First**, run `2DCNN.py` to prepare the data for the MDCNN model and train the 2D CNN model.
   
2. **Next**, run `1DCNN.py` to perform feature extraction and train the 1D CNN model.

3. **Then**, run `ROI_selection_energy_feature_extraction_analysis.py` for ROI selection and feature extraction analysis.

4. **Finally**, use `MVMDM.py` to train the 3D CNN model and test its performance and prediction of schizophrenia.

