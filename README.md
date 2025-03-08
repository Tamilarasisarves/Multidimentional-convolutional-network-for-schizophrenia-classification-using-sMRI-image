# Multidimentional-convolutional-network-for-schizophrenia-classification-using-sMRI-image
This repository contains Python scripts for classifying schizophrenia patients based on their structural MRI scans using a multidimensional convolutional network.
## Data Availability Statement

The datasets generated and/or analyzed during the current study are available in the following open repositories:

- **UCLA Consortium for Neuropsychiatric Phenomics LA5c Study database**: [https://openfmri.org/dataset/ds000030/](https://openfmri.org/dataset/ds000030/)
- **OpenfMRI dataset for schizophrenia**: [https://openfmri.org/dataset/ds000115/](https://openfmri.org/dataset/ds000115/)
- **COBRE dataset**: [https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html](https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html)

## How to Run:

1. **First**, run `2DCNN.py` to prepare the data for the MDCNN model and train the 2D CNN model.
   
2. **Next**, run `1DCNN.py` to perform feature extraction and train the 1D CNN model.

3. **Then**, run `ROI_selection_energy_feature_extraction_analysis.py` for ROI selection and feature extraction analysis.

4. **Finally**, use `MVMDM.py` to train the 3D CNN model and test its performance using the trained model.

