import numpy as np
from utils import min_max_normalize, z_score_normalize, euclidean_distance, manhattan_distance


class KNN:
    def __init__(self, k, percentage, quantity_classes, dataset_training, dataset_test, dataset_training_results, dataset_test_results, normalize_type='zscore', distance_type='euclidean'):
        self.k = k
        self.percentage = percentage
        self.quantity_classes = quantity_classes
        self.normalize_type = normalize_type
        self.distance_type = distance_type
        self.dataset_training = dataset_training
        self.dataset_test = dataset_test
        self.dataset_training_results = dataset_training_results
        self.dataset_test_results = dataset_test_results
        self.distance_array = []
        self.predictions = []

        if self.percentage != 1:
            for i in range(quantity_classes):
                quantity_element_class = int(len(dataset_training) / quantity_classes)
                indices = np.random.permutation(quantity_element_class)

                dataset_training_copy = self.dataset_training[i*quantity_element_class:(quantity_element_class*(i+1))]
                dataset_training_results_copy = self.dataset_training_results[i*quantity_element_class:(quantity_element_class*(i+1))]

                split_idx = int(quantity_element_class * self.percentage)

                train_idx, val_idx = indices[split_idx:], indices[:split_idx]

                if i == 0:
                    dataset_training_copy_final = dataset_training_copy[val_idx]
                    dataset_training_results_copy_final = dataset_training_results_copy[val_idx]
                else:
                    dataset_training_copy_final = np.concatenate((dataset_training_copy_final, dataset_training_copy[val_idx]))
                    dataset_training_results_copy_final = np.concatenate((dataset_training_results_copy_final, dataset_training_results_copy[val_idx]))

            self.dataset_training = dataset_training_copy_final
            self.dataset_training_results = dataset_training_results_copy_final

    def __normalize(self, type='zscore'):
        if type is not None:
            if type == 'minmax':
                self.dataset_training = min_max_normalize(self.dataset_training)
                self.dataset_test = min_max_normalize(self.dataset_test)

                return

            self.dataset_training = z_score_normalize(self.dataset_training)
            self.dataset_test = z_score_normalize(self.dataset_test)


    def __distance(self, type='euclidean'):
        if type == 'manhattan':
            for i in range(len(self.dataset_test)):
                for j in range(len(self.dataset_training)):
                    self.distance_array.append(manhattan_distance(self.dataset_test[i], self.dataset_training[j]))

        for i in range(len(self.dataset_test)):
            for j in range(len(self.dataset_training)):
                self.distance_array.append(euclidean_distance(self.dataset_test[i], self.dataset_training[j]))

    def __accuracy(self):
        return np.mean(self.predictions == self.dataset_test_results)


    def knn(self):
        self.__normalize(self.normalize_type)
        self.__distance(self.distance_type)

        for i in range(len(self.dataset_test)):
            k_indices = np.argsort(self.distance_array[i*len(self.dataset_training):(i+1) * len(self.dataset_training)])[:self.k]
            k_nearest_labels = [self.dataset_training_results[i] for i in k_indices]
            self.predictions.append(np.argmax(np.bincount(k_nearest_labels)))

        return self.__accuracy()
