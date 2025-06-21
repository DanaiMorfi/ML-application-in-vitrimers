# ML-application-in-vitrimers
Development and application of machine learning models for the prediction of critical properties of dynamically crosslinked polymers.

This GitHub contains all the code and excel files created and used during this work. 
Here, you can find all necessary information to better understand the workflow used, reproduce results or even build on the existing work and database and expand them to further research into this fascinating subject.

The logic behind the filing system is this: All files are numbered in a way that clearly indicates the order in which they should be used, along with their respective names. 
Because each code snippet is stored in its respective file, some documents (specifically excel files) need to be carefully copied and transferred from one file to the next in order for the next piece of code to work. Unifying all code snippets into one big code or arranging them in a single file was avoided, so as to keep everything as clean as possible. 

The basic functions behind every file will now be explained:

A. Excel files: Contains all excel files used and created during this work. This involves input excels like SMILES list excel ("SMILES_list") and curated database excel ("Draft4"), and output excels like SMILES list with pictures for visualization of molecules used during vitrimer synthesis ("SMILES"), model training results comparing between PCA and the alternative method ("dim_red_X", with X= d, Tg, Tv, gel) and results comparing between tuned and untuned models ("hp_tuning_X, with X= d, Tg, Tv, gel).

B. Coding files: Contains every piece of code used during this work, divided into steps.

B1. Feature extraction: Contains the script "rdkit_descr" which uses "SMILES_list" and extracts all available RDKit descriptors through RDKit functions and stores the results in the "SMILES_list_with_descriptors" excel.

B2. Pre-processing: Contains two separate files: one for data (experimental set) and one for descriptors (feature set). The first contains the "data_normilization_v1" script that reads "Draft4", applies pre-processing as described in the paper (missing value imputation, filling in empty cells with the No.molec. name and applying linear normalization) and saves cleaned results in "Draft4_cleaned2" and cleaned AND normalized results in "normalized_Draft42". It also creates a separate file containing histograms for all properties for overview reasons. The second contains two scripts: 1. "feat_normilization_clean" and 2. "skewness check". The first reads "SMILES_list_with_descriptors", applies processing as described in the paper (zero and low standard deviation filter and scarcity filter) and saves cleaned results in "SMILES_list_cleaned2". It also creates a file for histograms of all non-normalized features. The second reads "SMILES_list_cleaned2", applies normalization as described in the paper (applies linear normalization for non- skewed features and log normalization for skewed features) and saves results in "normalized_SMILES_list2". It also creates a file for histograms of all normalized features.

B3. Dimensionality reduction:

B4. Model training:

B5. Hyperparameter tuning:

B6. Plotting: 

