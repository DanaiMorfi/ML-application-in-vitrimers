# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 21:54:39 2025

@author: danai
"""

#This is a code that performs feature extraction using the RDKit library
#and SMILES identifiers collected from online resources for the various
#molecules used bibliographically

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load Excel file
data = pd.read_excel("SMILES_list.xlsx")

# Ensure the first column contains SMILES
smiles_list = data.iloc[:, 0].astype(str)

# Function to compute all RDKit descriptors
def get_rdkit_descriptors(mol):
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    descriptor_values = [desc[1](mol) if mol else None for desc in Descriptors.descList]
    return dict(zip(descriptor_names, descriptor_values))

# Process each SMILES
results = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        descs = get_rdkit_descriptors(mol)
        results.append(list(descs.values()))
    else:
        results.append([None] * len(Descriptors.descList))

# Create DataFrame with descriptor names
descriptor_names = [desc[0] for desc in Descriptors.descList]
descriptor_df = pd.DataFrame(results, columns=descriptor_names)

# Insert results after the 3rd column
data = pd.concat([data.iloc[:, :3], descriptor_df], axis=1)

# Save to Excel
data.to_excel("SMILES_list_with_descriptors.xlsx", index=False)

print("Descriptors added successfully!")
