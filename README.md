# ML-application-in-vitrimers
Development and application of machine learning models for the prediction of critical properties of dynamically crosslinked polymers.


This GitHub contains all the code and excel files created and used during this work. 
Here, you can find all necessary information to better understand the workflow used, reproduce results or even build on the existing work and database and expand them to further research into this fascinating subject.

The logic behind the filing system is this: All files are numbered in a way that clearly indicates the order in which they should be used, along with their respective names. 
Because each code snippet is stored in its respective file, some documents (specifically excel files) need to be carefully copied and transferred from one file to the next in order for the next piece of code to work. Unifying all code snippets into one big code or arranging them in a single file was avoided, so as to keep everything as clean as possible. 


The basic functions behind every file are as follows:

A. Excel files: Contains all excel files used and created during this work. This involves input excels like SMILES list excel ("SMILES_list") and curated database excel ("Draft4_new2"), cleaned curated database excel used during plotting ("Draft4_cleaned2") and output excels like SMILES list with pictures for visualization of molecules used during vitrimer synthesis ("SMILES_list_with_pictures"), model training results comparing between PCA and the alternative method ("dim_red_X", with X= d, Tg, Tv, gel) and results comparing between tuned and untuned models ("hp_tuning_X, with X= d, Tg, Tv, gel).


B. Coding files: Contains every piece of code used during this work, divided into steps.


B1. Feature extraction: Contains the script "rdkit_descr" which uses "SMILES_list" and extracts all available RDKit descriptors through RDKit functions and stores the results in the "SMILES_list_with_descriptors" excel.


B2. Pre-processing: Contains two separate files: one for data (experimental set) and one for descriptors (feature set). The first contains the "data_normilization_v1" script that reads "Draft4_new2", applies pre-processing as described in the paper (missing value imputation, filling in empty cells with the No.molec. name and linear normalization application) and saves cleaned results in "Draft4_cleaned2" and cleaned AND normalized results in "normalized_Draft42". It also creates a separate file containing histograms for all properties for overview reasons. The second contains two scripts: 1. "feat_normilization_clean" and 2. "skewness check". The first reads "SMILES_list_with_descriptors", applies processing as described in the paper (zero and low standard deviation filter and scarcity filter) and saves cleaned results in "SMILES_list_cleaned2". It also creates a file for histograms of all non-normalized features. The second reads "SMILES_list_cleaned2", applies normalization as described in the paper (applies linear normalization for non- skewed features and log normalization for skewed features) and saves results in "normalized_SMILES_list2". It also creates a file for histograms of all normalized features.


B3. Dimensionality reduction: Contains four files: one for each target variable so density, Tg, Tv, gel. Each contains two files: "alt" and "pca". The first contains several scripts like: 1. "cor_anal_data" which takes "normalized_Draft42", performs correlation analysis on it and saves the results in "normalized_Draft4_cleaned_final2"/ 2. "cor_anal_feat" that does the same for "normalized_SMILES_list2" and creates "normalized_SMILES_list_cleaned_final2"/ 3. "merge" that combines "normalized_Draft4_cleaned_final2" and "normalized_SMILES_list_cleaned_final2" into one united dataset named "augmented_output_clean2"/ 4. "last_cor" that performs correlation analysis on "augmented_output_clean2" and saves results in "augmented_output_clean1_final2"/ 5. "X_DR" where X= MI or RF that read "augmented_output_clean1_final2", train an MI or RF algorithm to perform feature selection and save results in "features_X"/ 6. "features_features_RF-MI" that read "augmented_output_clean1_final2", train both algorithms and save combined total results in "features_RF_MI_combined". The second file contains two scripts: 1. "merge" that does the same as before but saves results in "augmented_output_clean"/ 2. "pca" which takes "augmented_output_clean", trains a PCA algorithm, creates 4 principal components and saves results in "pca_transformed_features". It also creates a PCA heatmap.


B4. Model training: Contains four files like dimensionality reduction. Each contains two files: "alt" and "pca". The first contains the script "mod_train" that reads the "features_RF_MI_combined" excel, splits the data set using 5-fold cross validation, trains models using typical hyperparameters to predict each target variable, calculates validation metrics and saves results in "model_performance_summary". The second contains the same but instead of the "features_RF_MI_combined" excel, it uses the "pca_transformed_features" to train models.


B5. Hyperparameter tuning: Contains three files: "pca_d", "dim_Tg" and "dim_Tv". The first contains the script "mod_train_CV_d" that reads the "pca_transformed_features" excel, applies 5-fold cross-validation and grid search, trains models using the optimal hyperparameters, calculates validation metrics and saves results in "model_performance_summary_CV". The same applies for the other two but instead of "pca_transformed_features" they use "features_RF_MI_combined", changing the name of the main script accordingly.


B6. Plotting: Same as model training with extra features. The main script ("model_train_parity_X_X") creates parity plots and error distribution plots, using results from model training and true values from "Draft4_cleaned2" after reversing the normalization that was applied earlier. The files also contain the script "bar_charts" that create a box plot that compares metrics for all models predicting the same target variable. This is applied for all target variables for both dimensionality reduction methods, as well as the tuned models, using results from B5.

