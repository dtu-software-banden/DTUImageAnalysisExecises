import numpy as np
import pandas as pd

from utils.pca_utils import compute_pca, count_needed_to_explain,normalize,max_projected_abs_value

# Load the CSV file
# Assuming the file is named 'glass_data.csv' and uses whitespace as delimiter
df = pd.read_csv('exams/2023-fall-Thor/section5/pistachio_data.txt',sep='\s+')



colMean = np.mean(df,axis=0)
colStd = np.std(df,axis=0)
normDf = (df - colMean) / colStd

pca,projections = compute_pca(normDf,n_components=12)

nut1Proj = projections[0,:]
print("Distance:",np.sum(nut1Proj ** 2))

print("PCA count:",count_needed_to_explain(pca,0.97))


cov_matrix = np.cov(normDf, rowvar=False)

print("Max abs:",np.max(np.abs(cov_matrix)))

