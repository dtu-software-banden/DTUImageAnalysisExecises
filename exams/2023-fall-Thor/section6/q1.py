import numpy as np
from skimage import transform as tf


landmarks1 = np.array([(1,0),(5,0),(2,4),(4,4),(3,6)])
landmarks2 = np.array([(3,1),(7,1),(3.5,3),(5.5,5),(4.5,6)])

def sumSq (lm1,lm2):
    return np.sum((lm1 - lm2)**2)

print(sumSq(landmarks1,landmarks2))

cMin = sumSq(landmarks1,landmarks2)
minTrans = (0.,0.)

for dxi in range(-100,100):
    for dyi in range(-100,100):
        dx = dxi / 10.0
        dy = dyi / 10.0
        nlm2 = landmarks2 + np.array([(dx,dy),(dx,dy),(dx,dy),(dx,dy),(dx,dy)])
        newsq = sumSq(landmarks1,nlm2)
        if newsq < cMin:
            minTrans = (dx,dy)
            cMin = newsq

print("Min:",cMin,minTrans)

tform = tf.estimate_transform('similarity', landmarks2, landmarks1)

print(tform.rotation*180 / (2*3.14))
