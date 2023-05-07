import numpy as np


def transform_text_to_numpy(path):
    arrayNumpy = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip().split()
            arrayNumpy.append([float(x) for x in line])
    return np.array(arrayNumpy)


def min_max_normalize(dataset):
    min = np.min(dataset)
    max = np.max(dataset)

    return (dataset - min) / (max - min)


def z_score_normalize(dataset):
    media = np.mean(dataset, axis=0)
    standard_deviation = np.std(dataset, axis=0)

    return (dataset - media) / standard_deviation



def euclidean_distance(x, y):
    # Distance from x to y
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x, y):
    # Distance from x to y
    return np.sum(np.abs(x - y))
