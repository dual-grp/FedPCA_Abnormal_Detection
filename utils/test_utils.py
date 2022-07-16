'''
Import necessary libraries
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from utils.pca_utils import *

'''
Data preprocessing 
'''

MIN_MAX = 1
STANDAR = 2
ROBUST = 3

# Choose the preprocessing methods:
def prep_data(dataX, prep_type=1):

    change_dataX = dataX.copy()
    featuresToScale = change_dataX.columns

    if prep_type == MIN_MAX:
        min_max = MinMaxScaler()
        change_dataX.loc[:,featuresToScale] = min_max.fit_transform(change_dataX[featuresToScale])
    elif prep_type == STANDAR:
        sX = StandardScaler(copy=True)
        change_dataX.loc[:,featuresToScale] = sX.fit_transform(change_dataX[featuresToScale])
    else:
        robScaler = RobustScaler()
        change_dataX.loc[:,featuresToScale] = robScaler.fit_transform(change_dataX[featuresToScale])

    return change_dataX


'''
 Define the score function for abnormal detection
'''
def anomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF) - np.array(reducedDF))**2, axis=1) 
    return loss


'''
Function test on all types of attacks from csv file
'''
def test_on_random_attack_samples_csv(V_k, thres_hold = 6, prep_type=MIN_MAX):
    dataX = pd.read_csv('abnormal_detection_data/test/abnormal_test_34_fea.csv', index_col=False)
    dataX = dataX.drop(['outcome', 'Unnamed: 0'], axis=1)
    dataX = prep_data(dataX, prep_type=STANDAR)
    data_transform = self_pca_transform_with_zero_mean(dataX, V_k)
    data_inverse = self_inverse_transform_with_zero_mean(data_transform, V_k)
    dataX = prep_data(dataX, prep_type=MIN_MAX)
    data_inverse = prep_data(data_inverse, prep_type=MIN_MAX)
    abnormal_score = anomalyScores(dataX, data_inverse)
    index = abnormal_score < thres_hold
    num_samples = dataX.shape[0]
    total_misclassified_samples = len(abnormal_score[index])
    return num_samples, total_misclassified_samples

'''
Test on another 10000 sample of normal data get data from csv file
'''
def test_on_normal_data_samples_csv(V_k, thres_hold=6):
    normal_data_10000 = pd.read_csv('abnormal_detection_data/test/kdd_10000_34_fea.csv', index_col=False)
    # Standardization 
    normal_data_10000 = prep_data(normal_data_10000, prep_type=STANDAR)
    # Transform data using Federated PCA components
    normal_data_pca = self_pca_transform_with_zero_mean(normal_data_10000, V_k)
    normal_data_inverse = self_inverse_transform_with_zero_mean(normal_data_pca, V_k)
    # Min max normalization before testing to keep balance between features
    normal_data_10000 = prep_data(normal_data_10000, prep_type=MIN_MAX)
    normal_data_inverse = prep_data(normal_data_inverse, prep_type=MIN_MAX)
    # Get anomalous score
    abnormal_score = anomalyScores(normal_data_10000, normal_data_inverse)
    # Get the misclassified samples
    index = abnormal_score > thres_hold
    total_normal_samples = normal_data_10000.shape[0]
    return total_normal_samples, len(abnormal_score[index])
    
'''
Test on both normal and abnormal data and calculate the F-Score
'''
def kdd_test(V_k, thres_hold):
    # Get total samples and misclassified samples on normal data
    normal_total_samples, normal_mis_samples = test_on_normal_data_samples_csv(V_k, thres_hold=thres_hold)
    FN = normal_mis_samples
    TN = normal_total_samples - normal_mis_samples
    # Get total samples and misclassified samples on abnormal data
    abnormal_total_samples, abnormal_mis_samples = test_on_random_attack_samples_csv(V_k, thres_hold=thres_hold)
    FP = abnormal_mis_samples
    TP = abnormal_total_samples - abnormal_mis_samples
    # Estimate performance based on the F-Score
    precision_score = TP/(FP + TP)
    recall_score = TP/(FN + TP)
    accuracy_score = (TP + TN)/ (TP + FN + TN + FP)
    f1_score = 2*precision_score*recall_score/(precision_score + recall_score)
    print(f"Precision: {precision_score * 100.0}")
    print(f"Recall: {recall_score * 100.0}")
    print(f"Accuracy score: {accuracy_score * 100.0}")
    print(f"F1 score: {f1_score * 100.0}")
    return accuracy_score*100.0