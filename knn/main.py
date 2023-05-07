import numpy as np

from knn import KNN
from utils import transform_text_to_numpy, z_score_normalize

if __name__ == "__main__":
    # files = ['1x1', '2x2', '3x3', '5x5']
    # ks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    # distances = ['manhattan', 'euclidean']
    # percentages = [0.25, 0.50, 0.75, 1]
    #
    # dataset_training_results = np.repeat(np.arange(10), 100)  # Every 100 rows, the class changes [0, 1, 2, ..., 8, 9]
    # dataset_test_results = np.repeat(np.arange(10), 100)  # Every 100 rows, the class changes [0, 1, 2, ..., 8, 9]
    #
    # for file in files:
    #     print(f"Size: {file}")
    #
    #     file_name_training = f"dataset_student/treino_{file}.txt"
    #     file_name_test = f"dataset_student/teste_{file}.txt"
    #
    #     dataset_training = transform_text_to_numpy(file_name_training)
    #     dataset_test = transform_text_to_numpy(file_name_test)
    #
    #     for distance in distances:
    #         print(f"Distance: {distance}")
    #         for k in ks:
    #             print(f"K: {k}")
    #
    #             accuracies = []
    #             for percentage in percentages:
    #                 print(f"Percentagem: {percentage * 100}%")
    #
    #                 knn = KNN(k, percentage, 10, dataset_training, dataset_test, dataset_training_results, dataset_test_results, distance_type=distance, normalize_type='minmax')
    #                 accuracy = knn.knn()
    #                 print(f"Accuracy: {accuracy}")
    #
    #                 accuracies.append(accuracy)
    #
    #
    #             print(f"Average Accuracy: {np.mean(np.array(accuracies))}")
    #             print()
    #             print('---------------------------------')
    #             print()


    ks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    distances = ['manhattan', 'euclidean']
    percentages = [0.25, 0.50, 0.75, 1]

    dataset_training = transform_text_to_numpy('data_Train_Test/treinamento.txt')
    dataset_test = transform_text_to_numpy('data_Train_Test/teste.txt')

    dataset_training_results = dataset_training[:, -1].reshape(1, -1)[0]
    dataset_test_results = dataset_test[:, -1].reshape(1, -1)[0]

    dataset_training = dataset_training[:, :-1]
    dataset_test = dataset_test[:, :-1]

    for distance in distances:
        print(f"Distance: {distance}")
        for k in ks:
            print(f"K: {k}")

            accuracies = []
            for percentage in percentages:
                print(f"Percentagem: {percentage * 100}%")

                knn = KNN(k, percentage, 10, dataset_training, dataset_test, dataset_training_results, dataset_test_results, distance_type=distance, normalize_type='minmax')
                accuracy = knn.knn()
                print(f"Accuracy: {accuracy}")

                accuracies.append(accuracy)

            print(f"Average Accuracy: {np.mean(np.array(accuracies))}")
            print()
            print('---------------------------------')
            print()

