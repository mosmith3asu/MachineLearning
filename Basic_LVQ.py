# LVQ for the Ionosphere Dataset
from csv import reader
from random import randrange
from random import seed
seed(1)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.model_selection import train_test_split


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


class LVQ():
    def __init__(self, initial_data):
        self.prototype_size = np.size(initial_data[0])
        self.dataset = np.resize(initial_data, (-1, self.prototype_size))
        self.dataset_test = dataset
        self.verbose = True

        # Clustering parameters
        self.prototype_budget = 8  # (B) budget/max number of prototypes
        self.max_clusters = 5
        self.min_cluster_size = 3
        self.training_rounds = 5  # (I)
        self.M = np.size(np.unique(self.dataset[:, -1]))  # (M) number of different classes of labels

    # Evaluate an algorithm using a cross validation split
    def current_accuracy(self, dataset_test, prototypes=[]):
        if np.size(prototypes) < 1: prototypes = self.prototypes

        # Calculate predicted labels
        predictions = [self.get_best_matching_unit(prototypes, row)[-1] for row in dataset_test]
        # Isoalte true lables
        true_labels = np.reshape(dataset_test, (-1, self.prototype_size))[:, -1]

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
    def euclidean_distance(self, row1, row2):
        dist = 0.0
        row1 = np.array(row1)
        row2 = np.array(row2)
        for i in range(len(row1) - 1):
            dist += np.linalg.norm(row1 - row2)
        # distance.euclidean(row1[i],row2[i])
        return np.sqrt(dist)

    # Locate the best matching unit
    def get_best_matching_unit(self, codebooks, test_row):

        distances = list()
        for codebook in codebooks:
            dist = self.euclidean_distance(codebook, test_row)
            distances.append((codebook, dist))
        distances.sort(key=lambda tup: tup[1])
        return distances[0][0]

    # Make a prediction with codebook vectors
    def predict(self, test_row, prototypes=[]):
        if np.size(prototypes) < 1: prototypes = self.prototypes
        bmu = self.get_best_matching_unit(prototypes, test_row)
        return bmu[-1]

    # Create a random codebook vector
    def random_codebook(self, train):
        n_records = len(train)
        n_features = len(train[0])
        codebook = [train[randrange(n_records)][i] for i in range(n_features)]
        return codebook

    # Train a set of codebook vectors
    def train_codebooks(self, train, n_codebooks, lrate, epochs):
        codebooks = [self.random_codebook(train) for i in range(n_codebooks)]
        for epoch in range(epochs):
            rate = lrate * (1.0 - (epoch / float(epochs)))
            for row in train:
                bmu = self.get_best_matching_unit(codebooks, row)
                for i in range(len(row) - 1):
                    error = row[i] - bmu[i]
                    if bmu[-1] == row[-1]:
                        bmu[i] += rate * error
                    else:
                        bmu[i] -= rate * error
        print("trained")
        self.prototypes = np.reshape(codebooks, (-1, self.prototype_size))
        return codebooks

    def learning_vector_quantization(self, train, test, n_codebooks, lrate, epochs):
        codebooks = self.train_codebooks(train, n_codebooks, lrate, epochs)
        predictions = list()
        for row in test:
            output = self.predict(codebooks, row)
            predictions.append(output)
        print(np.size(predictions))
        return (predictions)

    # Clustering prototpye replacement
    # M classes
    # n_j misclassified points of class j
    # set bounds max clusters and min_cluster_size
    # new prototypes introduced at positions of cluster centrioids
    def Cluster_Replacement(self, train_data,prototypes):
        prototype_size = np.size(prototypes[0])
        train_data = np.reshape(train_data, (-1, prototype_size))

        P = np.reshape(prototypes, (-1, prototype_size))  # current prototypes
        #P = self.prototypes  # current prototypes
        Final_Prototypes = prototypes  # initialize final prototype variable
        N_train = np.size(train_data[:, 0])
        s = 0.6  # constant value between 0.4 and 0.8; "Window Rule" to prevent protoype from diverging

        max_it = self.training_rounds  # training rounds repeated I times unless accuracy stops improving
        it = 1  # iteration index
        t = 1  # how many updates have been made

        a0 = 0.8  # initial function value
        aT = 4. * N_train  # update step
        a_t = a0 * aT / (aT + t)  # monotonically decreasing function over time (2)

        err_sig = 0.015  # 10e-5 # error significance rate
        err0 = 1.0 - self.current_accuracy(train_data,prototypes=prototypes)
        err_it = [100., err0]
        print(err_it)
        AjBjCj = np.zeros((np.size(P[:, 0]), 3))

        while (it < max_it and (err_it[it] - err_it[it - 1] < err_sig)):# or it<3:

            if self.verbose: print("\n######### Iternation ", it, "#########")
            if self.verbose: print("Current Prototypes:\n", P)

            for i in range(N_train):
                xi, yi = train_data[i, :-1], train_data[i, -1]  # new training data point vector and class

                # find nearest prototype (may be train_data[i,:] instead of xi)
                pj = self.get_best_matching_unit(P, train_data[i,:])
                j = np.where(np.all(P == pj, axis=1))[0]  # index of nearest prototype
                #pj = self.get_best_matching_unit(self.prototypes, train_data[i,:])
                #j = np.where(np.all(self.prototypes == pj, axis=1))[0]  # index of nearest prototype
                mj, cj = pj[:-1], pj[-1]  # vector class split

                # find next best matching prototype
                pk = self.get_best_matching_unit(np.delete(P, j, 0),train_data[i, :])
                k = np.where(np.all(P == pk, axis=1))[0]  # index of second nearest prototype
                #pk = self.get_best_matching_unit(np.delete(self.prototypes, j, 0), train_data[i, :])
                #k = np.where(np.all(self.prototypes == pk, axis=1))[0]  # index of second nearest prototype
                mk, ck = pk[:-1], pk[-1]  # vector class split

                # If th nearest prototype misclassified and the second closest classified correctly
                if (cj != yi) and (ck == yi):
                    dj = self.euclidean_distance(np.append(xi,yi), np.append(mj,cj))  # find distance between nearest
                    dk = self.euclidean_distance(np.append(xi,yi), np.append(mk,ck))  # find distance between second nearest

                    if min(dj / dk, dk / dj) > s:
                        # update prototypes and scores
                        mj_updated = mj - a_t * (xi - mj)###############################
                        P[j, :] = np.append(mj_updated, cj)
                        AjBjCj[j, :] = AjBjCj[j, :] + [0, 1, 0]  # Bj++

                        mk_updated = mk + a_t * (xi - mk)
                        P[k, :] = np.append(mk_updated, ck)
                        AjBjCj[k, :] = AjBjCj[k, :] + [0, 0, 1]  # Ck++

                else:
                    AjBjCj[j, :] = AjBjCj[j, :] + [1, 0, 0]  # Aj++

                t = t + 1
                a_t = a0 * aT / (aT + t)

            it = it + 1
            # calculate classification error (err_it)
            err_it.append(1. - self.current_accuracy(self.dataset_test, P))
            #err_it.append(1. - self.current_accuracy(self.dataset_test, Final_Prototypes))

            if err_it[it] <= err_it[it - 1]:
                Final_Prototypes = P  # store current prototypes as final prototypes
                if self.verbose: print("Modified Codebook Stored...")
                if self.verbose: print(P)
            else:
                if self.verbose: print("Current Codebook NOT Stored...")

            if it != max_it:  # If not last iteration
                print('AjBjCj Updated:\n', AjBjCj)
                Aj, Bj, Cj = np.hsplit(AjBjCj, 3)
                P = self.LVQremove(Aj, Bj, Cj, P)  # remove prototypes with negative scores
                P = self.LVQadd(P, train_data)
                AjBjCj = np.zeros((np.size(P[:, 0]), 3))  # reinitialize scoring values

            np.random.shuffle(train_data)  # shuffle data for text iteration
            #it = it + 1
            if self.verbose: print('Error Current Iteration: ', err_it[it])
            if self.verbose: print("Error Previous Iteration", err_it[it - 1])
            if self.verbose: print('Change in Error: ', err_it[it] - err_it[it - 1])


        # ALGORITHM CONCLUDED. REPORT RESULTS
        if it > max_it:
            if self.verbose: print("\n\nAlgorithm Stop (Max iterations reached)...")
        if err_it[it] - err_it[it - 1] >= err_sig:
            if self.verbose: print("\n\nAlgorithm Stop (No significant change in error)...")

        if self.verbose: print("Number of Iterations=", it-1, "/", max_it)
        return Final_Prototypes  # if accuracy stops improving, stop algorithm

    def LVQadd(self, prototypes, test_data, cluster_method = "hierarchical"):
        print('Begin LVQadd..........')
        # U_j in U_p = misclassified points in class
        # C_j = centroids found for class j
        N_prototypes = np.size(prototypes[:, 0])
        N_unique_classes = len(np.unique(np.reshape(self.prototypes, (-1, self.prototype_size))[:, -1]))

        min_cluster = self.min_cluster_size
        max_clusters = self.max_clusters
        prototype_budget = self.prototype_budget

        U_j = [self.misclassified_points_in_class(prototypes,test_data, j) for j in range(N_unique_classes)]
        U_j = [j.reshape(-1,self.prototype_size)[:,:-1] for j in U_j]   # reformat

        if cluster_method == "hierarchical":
            ClusterObj_j = [AgglomerativeClustering(n_clusters=max_clusters, affinity='euclidean', linkage='ward').fit(j) for j in U_j]

            Centroids_j = []
            for j in range(len(ClusterObj_j)):

                labels = np.reshape(ClusterObj_j[j].labels_, (-1, 1))
                U_j_labeled = np.append(U_j[j],labels,axis=1)

                label_sizes = np.unique(labels, return_counts=True)[1]

                centroids = []
                for label in np.unique(labels): # label means clustering group/label
                    out_of_label = np.where(U_j_labeled[:,-1]!=label)
                    j_in_class = np.delete(U_j_labeled,out_of_label,0)
                    label_centroid = [np.mean(j_in_class[:,col]) for col in range(j_in_class.shape[1]-1)]
                    label_centroid.append(j)        # append the class label to centroids
                    label_centroid.append(label_sizes[label])   # append label size
                    centroids.append(label_centroid)

                centroids = np.reshape(centroids,(-1,len(centroids[0])))
                Centroids_j.append(centroids)

            AllCentroids = np.reshape(Centroids_j,(-1,np.shape(centroids)[1]))

        elif cluster_method == "Kmeans":
            ClusterObj_j = [KMeans(max_clusters).fit(j) for j in U_j]  # perform Kmeans clustering for each class
            Cluster_centers = [cluster.cluster_centers_ for cluster in ClusterObj_j]
            # print("Cluster_Centeres:\n", Cluster_centers)
            Cluster_sizes = [np.unique(cluster.labels_, return_counts=True)[1] for cluster in ClusterObj_j]
            # print('Cluster Sizes:\n',Cluster_sizes)

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

        # print('AllCentroids:\n',AllCentroids)
        # Eliminate clusters that do not meet minimum cluster size
        indexs = [i for i in range(len(AllCentroids)) if AllCentroids[i, -1] < min_cluster]
        AllCentroids = np.delete(AllCentroids, indexs, axis=0)
        # print('AllCentroids Over Min:\n',AllCentroids)

        AllCentroids = AllCentroids[np.argsort(AllCentroids[:, -1])]  # Sort all centroids
        print('Sorted AllCentroids over min_cluster size:\n', AllCentroids)
        AllCentroids = AllCentroids[:, :-1]  # remove cluster size from array

        while N_prototypes < prototype_budget:  # append new prototypes as long as we are in limit of budget
            print('Adding Prototype N=', N_prototypes, ':  ', np.array([AllCentroids[-1]]), )
            prototypes = np.append(prototypes, np.array([AllCentroids[-1]]),
                                   axis=0)  # Add the first centroid to existing protoypes
            # prototypes.append(AllCentroids[-1])   # Add the first centroid to existing protoypes
            # del AllCentroids[-1] # remove first centroid
            AllCentroids = np.delete(AllCentroids, -1, 0)
            N_prototypes = N_prototypes + 1

        return prototypes

    def LVQremove(self, A_j, B_j, C_j, prototypes):
        print('Begining LVQremove..........')
        # Detects prototypes whose removal would result in accuracy increase
        # A_j = how many times protype m_j classified correctly and was not moved
        # -B_j = how many times it was moved away as the prototype of the wrong class
        # C_j = how many times it was moved towards as the protoype of the correct class
        # Score_j = A_j - B_j + C_j
        # prototypes with negative scores are likely to be detrimental to accuracy and therefore removed

        Score_j = [(A_j[j] - B_j[j] + C_j[j])[0] for j in range(len(prototypes))]
        print("Prototype Scores:\n", Score_j)

        # find and remove protoypes with negative scores
        neg_j = [j for j in range(len(Score_j)) if Score_j[j] < 0]
        print('Prototypes with negative scores:\n', neg_j)
        # print(np.reshape(prototypes,(-1,self.prototype_size)))
        for j in neg_j:
            print('Removing Prototype N=', j, ': ', prototypes[j, :])
        new_prototypes = np.delete(prototypes, neg_j, axis=0)

        return new_prototypes

    def misclassified_points_in_class(self,prototypes, test_dataset, class_index):
        predictions = [self.predict(row,prototypes=prototypes) for row in test_dataset]
        true_labels = np.reshape(test_dataset, (-1, self.prototype_size))[:, -1]

        misclassified_points = []
        for i in range(len(predictions)):
            # if prediction does not match true label (Misclassified) and prediction matches designated class
            if predictions[i] != true_labels[i] and predictions[i] == class_index:
                if np.size(misclassified_points) < 1:  # if not initialized
                    misclassified_points = dataset[i]  # then initialize array
                else:  # else append row
                    misclassified_points = np.append(misclassified_points, dataset[i], axis=0)

        return misclassified_points



# Test LVQ on Ionosphere dataset


# load and prepare data
filename = 'data_banknote_authentication2d.csv'
# filename = 'california_housing_test.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)

dataset_train, dataset_test = train_test_split(dataset, test_size=0.30, random_state=2)
dataset_train1, dataset_train2 = train_test_split(dataset_train, test_size=0.7, random_state=2)
# dataset_train= dataset
# dataset_test = dataset

print("Origonal Dataset Shape:", np.shape(dataset))
print("Training Dataset Shape:", np.shape(dataset_train))
print("Unused Dataset Shape:", np.shape(dataset_test))

# evaluate algorithm
n_folds = 5
learn_rate = 0.3
n_epochs = 50
n_codebooks = 6

LVQ = LVQ(dataset_train1)
LVQ.dataset_test = dataset_test
prototypes = LVQ.train_codebooks(dataset_train1, n_codebooks, learn_rate, n_epochs)
prev_acc = LVQ.current_accuracy(dataset_test)
prev_prototypes = prototypes

print("\nBegininng Cluster Replacment...\n")
prototypes = LVQ.Cluster_Replacement(dataset_train2,prev_prototypes)
acc = LVQ.current_accuracy(prototypes,dataset_test)

print("\nInitial Prototypes:\n", np.reshape(prev_prototypes, (-1, LVQ.prototype_size)))
print('\nInitial Score: %s' % prev_acc)
print("\nNew Prototypes:\n", prototypes)
# Score algorithm

print('\nScores: %s' % acc)

PLOT_ENABLE = True
if PLOT_ENABLE:
    plot_dim = 2
    fig = plt.figure()
    colors = ["blue", "green", "purple"]
    prototype_marker_size = 50

    if plot_dim == 3:
        ax = plt.axes(projection='3d')
        # Load a CSV file

        # Plot dataset
        plot_data = np.reshape(dataset, (-1, 4))
        for x, y, z, label in plot_data:
            color = colors[int(label)]
            ax.scatter(x, y, z, color=color, alpha=0.5, s=1)

        # Plot prototypes
        for x, y, z, label in prototypes:
            color = colors[int(label)]
            ax.scatter(x, y, z, color=color, marker="^", s=prototype_marker_size)


    elif plot_dim == 2:
        ax = plt.axes()
        # Load a CSV file

        # Plot dataset
        plot_data = np.reshape(dataset, (-1, LVQ.prototype_size))
        for x, y, label in plot_data:
            color = colors[int(label)]
            ax.scatter(x, y, color=color, alpha=0.5, s=1)

        # Plot prototypes
        for x, y, label in prototypes:
            color = colors[int(label)]
            ax.scatter(x, y, color=color, marker="*", s=prototype_marker_size)

        # Plot initial prototypes
        for x, y, label in prev_prototypes:
            color = colors[int(label)]
            ax.scatter(x, y, edgecolors=color, marker='o', facecolors='none', s=prototype_marker_size * 1.2)

        misclassified_class = [LVQ.misclassified_points_in_class(prototypes,plot_data, j) for j in np.unique(plot_data[:, -1])]
        misclassified_pts = misclassified_class[0]
        misclassified_pts = np.append(misclassified_pts, misclassified_class[1], axis=0)
        misclassified_pts = np.reshape(misclassified_pts, (-1, LVQ.prototype_size))

        for x, y, label in misclassified_pts:
            color = 'red'
            ax.scatter(x, y, color=color, alpha=0.5, s=3)

        ax.legend(['Data', 'O - Initial Prototoypes', '* - Replaced Prototypes', 'Misclassified Points'])

    plt.show()
