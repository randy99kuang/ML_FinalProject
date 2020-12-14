import sklearn.decomposition
from imports import *


def pca_adv_detector(x_train, input, threshold):
    train_reshaped = np.reshape(x_train, (-1, 28 * 28))
    input_reshaped = np.reshape(input, (-1, 28 * 28))
    pca = sklearn.decomposition.PCA()
    pca.fit(train_reshaped)
    pca.fit(train_reshaped)

    pca_score_train = pca.transform(train_reshaped)
    pca_score_input = pca.transform(input_reshaped)
    # print(pca_score_train.shape)
    # print(pca_score_input.shape)

    avg_score_train = np.mean(np.abs(pca_score_train), axis=0)
    # print(avg_score_train.shape)
    score_input = np.abs(pca_score_input)
    # print(score_input.shape)
    adv_tag = threshold_detect(avg_score_train, score_input, 0.008)
    return adv_tag


def threshold_detect(train_score, input_score, thresh):
    train_score_high_pc = train_score[-51:-1].reshape((1, 50))
    # print(train_score_high_pc.shape)
    input_score_high_pc = input_score[:, -51:-1]
    # print(input_score_high_pc.shape)
    diff = np.mean(input_score_high_pc - train_score_high_pc, axis=1)
    tag = (diff > thresh).astype(int)
    return tag