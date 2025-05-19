from utils.classifier_utils import parametric_classifier_predict, threshold_min_dist_classification


cows = [26, 46, 33, 23, 35, 28, 21, 30, 38, 43]
sheep = [67, 27, 40, 60, 39, 45, 27, 67, 43, 50, 37, 100]


min_dist_thresh = threshold_min_dist_classification(cows,sheep)

print("Min dist thresh:",min_dist_thresh)


cowPred = parametric_classifier_predict(38,cows)
sheepPred = parametric_classifier_predict(38,sheep)
print("Cow:",cowPred,"Sheep",sheepPred)