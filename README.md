The data processing and model training workflow are structured as follows:

Data Pre-processing: The initial data processing is performed in final_code_carbonexport_flux.py. This script handles the binning and cleaning of the raw data, and the resulting processed datasets are saved as .mat files.

Model Training and Prediction: These .mat files are loaded into the training pipeline to develop the Random Forest and Gaussian Process Regression models. Once trained, these models generate the global three-dimensional predictions for the particle size distribution (PSD) and biovolume. 

Visualization: The resulting predictions are subsequently used as the input for all global mapping and depth-profile visualizations presented in this study, the plotting functions are in the second part of final_code_carbonexport_flux.py.

Data Availability: The original UVP5 observational datasets is available on request or can be directly downloaded from the reference Kiko et al..
