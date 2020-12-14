from csv import reader
from random import randrange
from random import seed

seed(1)
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split


# Evaluate an algorithm using a cross validation split
def current_accuracy(prototypes, dataset_test):
    # Calculate predicted labels
    prototype_size = np.size(prototypes[0])
    predictions = [get_best_matching_unit(prototypes, row)[-1] for row in dataset_test]
    # Isoalte true lables
    true_labels = np.reshape(dataset_test, (-1, prototype_size))[:, -1]

    # Calculate accuracy
    correct = 0
    misclassified_j = []
    for i in range(len(predictions)):
        if predictions[i] == true_labels[i]:
            correct = correct + 1
        else:
            misclassified_j.append(i)

    # if self.verbose: print("Misclassified Points during Testing:", len(misclassified_j))

    accuracy = correct / len(predictions)
    return accuracy


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    dist = 0.0
    row1 = np.array(row1)
    row2 = np.array(row2)
    for i in range(len(row1) - 1):
        dist += np.linalg.norm(row1 - row2)
    # distance.euclidean(row1[i],row2[i])
    return np.sqrt(dist)

    # Locate the best matching unit


def get_best_matching_unit(codebooks, test_row):
    distances = list()
    for codebook in codebooks:
        dist = euclidean_distance(codebook, test_row)
        distances.append((codebook, dist))
    distances.sort(key=lambda tup: tup[1])
    return distances[0][0]

    # Make a prediction with codebook vectors


def predict(test_row, prototypes):
    bmu = get_best_matching_unit(prototypes, test_row)
    return bmu[-1]

    # Create a random codebook vector


def random_codebook(train):
    n_records = len(train)
    n_features = len(train[0])
    codebook = [train[randrange(n_records)][i] for i in range(n_features)]
    return codebook

    # Train a set of codebook vectors


def train_codebooks(train, n_codebooks, lrate, epochs):
    codebooks = [random_codebook(train) for i in range(n_codebooks)]
    for epoch in range(epochs):
        rate = lrate * (1.0 - (epoch / float(epochs)))
        for row in train:
            bmu = get_best_matching_unit(codebooks, row)
            for i in range(len(row) - 1):
                error = row[i] - bmu[i]
                if bmu[-1] == row[-1]:
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error
    print("trained")
    prototypes = codebooks
    prototype_size = np.size(prototypes[0])
    prototypes = np.reshape(prototypes, (-1, prototype_size))
    return prototypes


def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
    codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
    predictions = list()
    for row in test:
        output = predict(codebooks, row)
        predictions.append(output)
    print(np.size(predictions))
    return (predictions)

    # Clustering prototpye replacement
    # M classes
    # n_j misclassified points of class j
    # set bounds max clusters and min_cluster_size
    # new prototypes introduced at positions of cluster centrioids


def Cluster_Replacement(current_prototypes, train_data,
                        verbose=True, max_iterations=5,budget=8):
    if verbose: print("\nBegininng Cluster Replacment...\n")
    prototypes = current_prototypes
    prototype_size = np.size(prototypes[0])

    # Reshape data for safety
    prototypes = np.reshape(prototypes, (-1, prototype_size))
    train_data = np.reshape(train_data, (-1, prototype_size))

    P = prototypes  # Varaible for dynamic prototypes
    Final_Prototypes = prototypes  # Variable for final prototypes returned
    N_train = np.size(train_data[:, 0])  # number of training vectors

    max_it = max_iterations  # training rounds repeated I times unless accuracy stops improving
    it = 1  # iteration index
    t = 1  # how many updates have been made

    a0 = 0.8  # initial function value
    aT = 4. * N_train  # update step
    a_t = a0 * aT / (aT + t)  # monotonically decreasing function over time (2)
    s = 0.6  # constant value between 0.4 and 0.8; "Window Rule" to prevent protoype from diverging

    err_sig = 0.015  # 10e-5 # error significance rate
    err0 = 1.0 - current_accuracy(prototypes, train_data)  # initialize error parameter
    err_it = [100., err0]  # initialize running error list

    AjBjCj = np.zeros((np.size(P[:, 0]), 3))  # initialize scoring matrix

    while (it < max_it and (err_it[it] - err_it[it - 1] < err_sig)):  # or it<3:

        if verbose: print("\n######### Iternation ", it, "#########")
        if verbose: print("Current Prototypes:\n", P)

        for i in range(N_train):
            xi, yi = train_data[i, :-1], train_data[i, -1]  # new training data point vector and class

            # find nearest prototype (may be train_data[i,:] instead of xi)
            pj = get_best_matching_unit(P, train_data[i, :])
            j = np.where(np.all(P == pj, axis=1))[0]  # index of nearest prototype
            mj, cj = pj[:-1], pj[-1]  # vector class split

            # find next best matching prototype
            pk = get_best_matching_unit(np.delete(P, j, 0), train_data[i, :])
            k = np.where(np.all(P == pk, axis=1))[0]  # index of second nearest prototype
            mk, ck = pk[:-1], pk[-1]  # vector class split

            # If the nearest prototype misclassified and the second closest classified correctly
            if (cj != yi) and (ck == yi):
                dj = euclidean_distance(np.append(xi, yi), np.append(mj, cj))  # find distance between nearest
                dk = euclidean_distance(np.append(xi, yi), np.append(mk, ck))  # find distance between second nearest

                if min(dj / dk, dk / dj) > s:
                    # update prototypes and scores
                    mj_updated = mj - a_t * (xi - mj)  # mj(t+1)
                    P[j, :] = np.append(mj_updated, cj)  # update to dynamic prototype
                    AjBjCj[j, :] = AjBjCj[j, :] + [0, 1, 0]  # Bj++

                    mk_updated = mk + a_t * (xi - mk)  # mk(t+1)
                    P[k, :] = np.append(mk_updated, ck)  # update to dynamic prototype
                    AjBjCj[k, :] = AjBjCj[k, :] + [0, 0, 1]  # Ck++
            else:
                AjBjCj[j, :] = AjBjCj[j, :] + [1, 0, 0]  # Aj++

            t = t + 1
            a_t = a0 * aT / (aT + t)

        it = it + 1  # increment algorithm iteration
        err_it.append(1. - current_accuracy(P, train_data))  # calculate classification error

        if err_it[it] <= err_it[it - 1]:  # if classificaiton error decreased
            Final_Prototypes = P  # store current prototypes as final prototypes
            if verbose: print("Modified Codebook Stored...")
            if verbose: print(P)
        else:
            if verbose: print("Current Codebook NOT Stored...")

        if it != max_it:  # If not last iteration
            print('AjBjCj Updated:\n', AjBjCj)
            Aj, Bj, Cj = np.hsplit(AjBjCj, 3)
            P = LVQremove(P, Aj, Bj, Cj)  # remove prototypes with negative scores
            P = LVQadd(P, train_data,budget=budget)  # Add new prototypes around centroids of misclassified points
            AjBjCj = np.zeros((np.size(P[:, 0]), 3))  # reinitialize scoring values

        np.random.shuffle(train_data)  # shuffle data for text iteration
        if verbose: print('Error Current Iteration: ', err_it[it])
        if verbose: print("Error Previous Iteration", err_it[it - 1])
        if verbose: print('Change in Error: ', err_it[it] - err_it[it - 1])

    # ALGORITHM CONCLUDED. REPORT RESULTS
    if it > max_it:
        if verbose: print("\n\nAlgorithm Stop (Max iterations reached)...")
    if err_it[it] - err_it[it - 1] >= err_sig:
        if verbose: print("\n\nAlgorithm Stop (No significant change in error)...")

    if verbose: print("Number of Iterations=", it - 1, "/", max_it)
    return Final_Prototypes  # if accuracy stops improving, stop algorithm


def LVQadd(prototypes, test_data,
           cluster_method="hierarchical", min_cluster_size=3, max_clusters=5,
           prototype_budget=8, verbose=True):
    if verbose: print('Begin LVQadd..........')
    # U_j in U_p = misclassified points in class
    # C_j = centroids found for class j
    prototype_size = np.size(prototypes[0])

    N_prototypes = np.size(prototypes[:, 0])
    N_unique_classes = len(np.unique(np.reshape(prototypes, (-1, prototype_size))[:, -1]))

    min_cluster = min_cluster_size

    U_j = [misclassified_points_in_class(prototypes, test_data, j) for j in range(N_unique_classes)]
    U_j = [np.reshape(j, (-1, prototype_size))[:, :-1] for j in U_j]  # reformat

    if cluster_method == "hierarchical":
        ClusterObj_j = [AgglomerativeClustering(n_clusters=max_clusters, affinity='euclidean', linkage='ward').fit(j)
                        for j in U_j if np.size(j) > prototype_size]

        Centroids_j = []
        for j in range(len(ClusterObj_j)):

            labels = np.reshape(ClusterObj_j[j].labels_, (-1, 1))
            U_j_labeled = np.append(U_j[j], labels, axis=1)

            label_sizes = np.unique(labels, return_counts=True)[1]

            centroids = []
            for label in np.unique(labels):  # label means clustering group/label
                out_of_label = np.where(U_j_labeled[:, -1] != label)
                j_in_class = np.delete(U_j_labeled, out_of_label, 0)
                label_centroid = [np.mean(j_in_class[:, col]) for col in range(j_in_class.shape[1] - 1)]
                label_centroid.append(j)  # append the class label to centroids
                label_centroid.append(label_sizes[label])  # append label size
                centroids.append(label_centroid)

            centroids = np.reshape(centroids, (-1, len(centroids[0])))
            Centroids_j.append(centroids)

        AllCentroids = np.reshape(Centroids_j, (-1, np.shape(centroids)[1]))

    elif cluster_method == "Kmeans":
        ClusterObj_j = [KMeans(max_clusters).fit(j) for j in U_j]  # perform Kmeans clustering for each class
        Cluster_centers = [cluster.cluster_centers_ for cluster in ClusterObj_j]
        Cluster_sizes = [np.unique(cluster.labels_, return_counts=True)[1] for cluster in ClusterObj_j]

        # Append all centroids of [xi_misclassified] with [class, cluster_size] into single array
        AllCentroids = []
        for j in range(len(Cluster_centers)):
            class_labels = np.full((Cluster_centers[j][:, 0].size, 1), j)
            cluster_sizes = Cluster_sizes[j].reshape((np.size(Cluster_sizes[j]), -1))
            # print('Sizes',cluster_sizes)
            Clusters_j = np.append(Cluster_centers[j], class_labels, axis=1)
            Clusters_j = np.append(Clusters_j, cluster_sizes, axis=1)
            if np.size(AllCentroids) < 1:
                AllCentroids = Clusters_j
            else:
                AllCentroids = np.append(AllCentroids, Clusters_j, axis=0)

    # Eliminate clusters that do not meet minimum cluster size
    indexs = [i for i in range(len(AllCentroids)) if AllCentroids[i, -1] < min_cluster]
    AllCentroids = np.delete(AllCentroids, indexs, axis=0)
    # print('AllCentroids Over Min:\n',AllCentroids)

    AllCentroids = AllCentroids[np.argsort(AllCentroids[:, -1])]  # Sort all centroids
    if verbose: print('Sorted AllCentroids over min_cluster size:\n', AllCentroids)
    AllCentroids = AllCentroids[:, :-1]  # remove cluster size from array

    while N_prototypes < prototype_budget:  # append new prototypes as long as we are in limit of budget
        if verbose: print('Adding Prototype N=', N_prototypes, ':  ', np.array([AllCentroids[-1]]), )
        prototypes = np.append(prototypes, np.array([AllCentroids[-1]]),
                               axis=0)  # Add the first centroid to existing protoypes
        AllCentroids = np.delete(AllCentroids, -1, 0)  # Delete that prototype from list
        N_prototypes = N_prototypes + 1  # update # of current prototypes

    return prototypes


def LVQremove(prototypes, A_j, B_j, C_j, verbose=True):
    print('Begining LVQremove..........')
    # Detects prototypes whose removal would result in accuracy increase
    # A_j = how many times protype m_j classified correctly and was not moved
    # -B_j = how many times it was moved away as the prototype of the wrong class
    # C_j = how many times it was moved towards as the protoype of the correct class
    # Score_j = A_j - B_j + C_j
    # prototypes with negative scores are likely to be detrimental to accuracy and therefore removed

    Score_j = [(A_j[j] - B_j[j] + C_j[j])[0] for j in range(len(prototypes))]
    if verbose: print("Prototype Scores:\n", Score_j)

    # find and remove protoypes with negative scores
    neg_j = [j for j in range(len(Score_j)) if Score_j[j] < 0]
    if verbose: print('Prototypes with negative scores:\n', neg_j)
    for j in neg_j:
        if verbose: print('Removing Prototype N=', j, ': ', prototypes[j, :])
    new_prototypes = np.delete(prototypes, neg_j, axis=0)

    return new_prototypes


def misclassified_points_in_class(prototypes, test_dataset, class_index):
    prototype_size = np.size(prototypes[0])
    predictions = [predict(row, prototypes) for row in test_dataset]
    true_labels = np.reshape(test_dataset, (-1, prototype_size))[:, -1]

    misclassified_points = []
    for i in range(len(predictions)):
        # if prediction does not match true label (Misclassified) and prediction matches designated class
        if predictions[i] != true_labels[i] and predictions[i] == class_index:
            if np.size(misclassified_points) < 1:  # if not initialized
                misclassified_points = test_dataset[i]  # then initialize array
            else:  # else append row
                misclassified_points = np.append(misclassified_points, test_dataset[i], axis=0)

    return misclassified_points