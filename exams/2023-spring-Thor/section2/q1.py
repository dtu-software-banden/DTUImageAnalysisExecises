import numpy as np
import pandas as pd

from exam_template.utils.pca_utils import compute_pca,normalize,max_projected_abs_value

# Load the CSV file
# Assuming the file is named 'glass_data.csv' and uses whitespace as delimiter
df = pd.read_csv('exams/2023-spring-Thor/section2/glass_data.txt',sep='\s+')

df_norm = normalize(df)

pca,projections = compute_pca(df_norm)

print("Top 3 ratio:",pca.explained_variance_ratio_[:3].sum())

cov_matrix = np.cov(df_norm, rowvar=False)

print("CovMat 0,0: ",cov_matrix[0,0])

print("So first:", df_norm['Na'][0])

print("Max projected value:",max_projected_abs_value(projections))