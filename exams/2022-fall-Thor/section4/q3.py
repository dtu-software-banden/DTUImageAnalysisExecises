import numpy as np

from utils.lda_utils import lda_predict_batch


covar = np.array([[2,0],[0,2]])

data1_mean = np.array([24,3])
data2_mean = np.array([30,7])


x = np.array([23,5])


res = lda_predict_batch(x,data1_mean,data2_mean,covar)
print(res)