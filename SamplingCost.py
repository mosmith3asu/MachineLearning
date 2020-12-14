import random

import numpy as np
from scipy.spatial import distance
from csv import reader
from random import randrange
from random import seed

seed(1)
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split

def squared_euclidean(x, w):
    x = np.ndarray.flatten(x)
    w = np.ndarray.flatten(w)
    dst = distance.euclidean(x, w)
    return dst**2


classes = []
prototypes = []
psi = []
error_count = 0


def mu(d1, d2):
    mu_ = (np.subtract(d1, d2))/(np.add(d1, d2))
    return mu_


def deriv(mu_, t=2):
    z = (t*(np.exp(-t*mu_)))/((1 + np.exp(-t*mu_))**2)
    return z


def get_best_matching_unit(prototype_list, sample):
    distances = []
    for prototype in prototype_list:
        dist = squared_euclidean(prototype[0], sample)
        distances.append((prototype, dist))
    # print(distances)
    distances.sort(key=lambda tup: tup[1])
    # print("\n\n", distances, type(distances), distances[0], type(distances[0]))
    print("Predicted Class: \n", distances[0][0][1])
    return distances[0][0][1]


def w_updates(class_of_image, feature, alpha=0.01):
    current_w1 = 0
    current_w2 = 0
    index_w1 = 0
    index_w2 = 0

    lowest_distance_same_class = float('inf')
    for i in range(len(prototypes)):
        if class_of_image in prototypes[i]:
            # print(prototypes[i])
            temp_lowest_same_class = squared_euclidean(feature, prototypes[i][0])
            # print("templowestsamplesame is {}".format(temp_lowest_same_class))
            if temp_lowest_same_class <= lowest_distance_same_class:
                lowest_distance_same_class = temp_lowest_same_class
                current_w1 = prototypes[i][0]
                index_w1 = i
                # print("\n Index w1 {} ".format(index_w1))

    d1 = lowest_distance_same_class

    list_w2 = [item for item in prototypes if class_of_image not in item]
    # print(len(list_w2), list_w2)
    if len(list_w2) == 0:
        d1 = 0
        d2 = 0
        return d1, d2

    lowest_distance_different_class = float('inf')
    for j in range(len(prototypes)):
        if class_of_image not in prototypes[j]:
            # print(prototypes[j])
            temp_lowest_different_class = squared_euclidean(feature, prototypes[j][0])
            # print("templowestsampledifferent is {}".format(temp_lowest_different_class))
            if temp_lowest_different_class <= lowest_distance_different_class:
                lowest_distance_different_class = temp_lowest_different_class
                current_w1 = prototypes[j][0]
                index_w2 = j
                # print("\n Index w2 {} ".format(index_w2))

    d2 = lowest_distance_different_class

    mu_ = mu(d1, d2)
    z = deriv(mu_)

    w1 = current_w1 + (alpha*z*4*d2*(feature - current_w1))/((np.add(d1, d2))**2)
    w2 = current_w2 - (alpha*z*4*d1*(feature - current_w2))/((np.add(d1, d2))**2)
    class_w1 = prototypes[index_w1][1]
    # print("class w1 is {}".format(class_w1))
    class_w2 = prototypes[index_w2][1]
    # print("class w2 is {}".format(class_w2))
    prototypes[index_w1] = (w1, class_w1)
    prototypes[index_w2] = (w2, class_w2)
    return d1, d2


def SamplingCost(psi, num_samples=10):
    print("\n Running SamplingCost\n")
    # print("\n Len psi is {}".format(len(psi)), '\n')
    if len(psi) < num_samples:
        num_samples = len(psi)

    # print(num_samples, '\n')
    minCost = float('inf')
    new_prototype = None
    psi_random = random.sample(psi, num_samples)
    # print("Random samples are \n {}".format(psi_random), len(psi_random))
    for ran_sam in range(num_samples):
        # print("\n RANSAM \n", ran_sam)
        (x, y) = (psi_random[ran_sam][0], psi_random[ran_sam][1])
        # print("XY is {} \n ".format((x, y)))
        psi_temp = updateShortTermMemory(psi, (x, y))
        calc_cost = calculateCost(psi_temp)
        if calc_cost < minCost:
            minCost = calc_cost
            new_prototype = (x, y)

    return new_prototype


def calculateCost(psi_temp):
    print("\n Calculating cost \n")
    cost = 0
    for i in range(len(psi_temp)):
        temp_mu = mu(psi_temp[i][2], psi_temp[i][3])
        temp_deriv = deriv(temp_mu)
        cost += temp_deriv
    return cost


def updateShortTermMemory(psi, x_y):
    print("\n Updating Short term memory \n")
    psi_temp = psi
    for iter_d in range(len(psi)):
        if psi_temp[iter_d][1] == x_y[1]:
            d_plus = squared_euclidean(psi_temp[iter_d][0], x_y[0])
            if psi_temp[iter_d][2] > d_plus:
                temp_x, temp_y, temp_dminus = psi_temp[iter_d][0], psi_temp[iter_d][1], psi_temp[iter_d][3]
                psi_temp[iter_d] = (temp_x, temp_y, d_plus, temp_dminus)

        if psi_temp[iter_d][1] != x_y[1]:
            # print("\n Going in is {} and {} \n".format(psi_temp[iter_d][0], x_y[0]))
            d_minus = squared_euclidean(psi_temp[iter_d][0], x_y[0])
            if psi_temp[iter_d][3] > d_minus:
                temp_x, temp_y, temp_dplus = psi_temp[iter_d][0], psi_temp[iter_d][1], psi_temp[iter_d][2]
                psi_temp[iter_d] = (temp_x, temp_y, temp_dplus, d_minus)

    return psi_temp
