import pickle
import warnings
import numpy as np
import random
import os
from gensim.models import KeyedVectors
import scipy.spatial.distance as distance
from sklearn import preprocessing


def generate_label_map():
    relation_map = dict()
    reverse_relation_map = dict()
    file = open('rel_map_' + dataset_name + '.txt')
    while 1:
        line = file.readline()
        if not line:
            break
        line = line.replace('\n', '')
        str = line.split('\t')
        relation = str[0]
        i = int(str[1])
        relation_map[relation] = i
        reverse_relation_map[i] = relation
    return relation_map, reverse_relation_map


def load_training_set_with_dist(path):
    dataset = list()
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        line = line.replace('\n', '')
        str = line.split('\t')
        first = str[0]
        second = str[1]
        relation = str[2]
        dist = np.zeros(shape=(relation_number))
        for i in range(0, relation_number):
            if reverse_relation_map[i] == relation:
                dist[i] = 1
            else:
                dist[i] = 0
        dataset.append((first, second, relation, dist))
    file.close()
    return dataset


def load_testing_set_with_dist(path):
    dataset = list()
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        line = line.replace('\n', '')
        str = line.split('\t')
        first = str[0]
        second = str[1]
        relation = str[2]
        dist = np.zeros(shape=(relation_number))
        for i in range(0, relation_number):
            dist[i] = float(str[i + 3])
        dataset.append((first, second, relation, dist))
    file.close()
    return dataset


def merge_datasets(training_set, testing_set):
    merged_dataset = list()
    for first, second, relation, dist in training_set:
        merged_dataset.append((first, second, relation, dist, True))
    for first, second, relation, dist in testing_set:
        merged_dataset.append((first, second, relation, dist, False))
    return merged_dataset


def get_edge_value(first_node, second_node):
    first_1, second_1, relation_1, dist_1, is_train_1 = first_node
    first_2, second_2, relation_2, dist_2, is_train_2 = second_node
    # training set prunning
    if is_train_1 and is_train_2:
        # both training
        if relation_1 == relation_2:
            return 1
        else:
            return 0
    # first train second test
    elif is_train_1 and is_train_2 == False:
        edge_value = 0
        train_relation_name = relation_1
        train_relation_id = relation_map[relation_1]
        W_matrix = models[train_relation_name]
        P_matrix_1 = (
            np.dot(W_matrix, en_model[first_1].reshape(n_embeddings, 1)) - en_model[second_1].reshape(n_embeddings,
                                                                                                      1)).T
        P_matrix_2 = (
            np.dot(W_matrix, en_model[first_2].reshape(n_embeddings, 1)) - en_model[second_2].reshape(n_embeddings,
                                                                                                      1)).T
        edge_value = (1 - distance.cosine(P_matrix_1, P_matrix_2)) * dist_2[train_relation_id]
        return edge_value * gamma
    elif is_train_1 == False and is_train_2:
        return get_edge_value(second_node, first_node)
    else:
        edge_value = 0
        for i in range(0, relation_number):
            relation_name = reverse_relation_map[i]
            W_matrix = models[relation_name]
            # first_representation
            P_matrix_1 = (
                np.dot(W_matrix, en_model[first_1].reshape(n_embeddings, 1)) - en_model[second_1].reshape(n_embeddings,
                                                                                                          1)).T
            # second_representation
            P_matrix_2 = (
                np.dot(W_matrix, en_model[first_2].reshape(n_embeddings, 1)) - en_model[second_2].reshape(n_embeddings,
                                                                                                          1)).T
            sim = (1 - distance.cosine(P_matrix_1, P_matrix_2)) * dist_1[i] * dist_2[i]
            edge_value = edge_value + sim
        return edge_value * gamma * gamma


def select_end_node(merged_dataset, sample_size, start_node):
    # select ends
    end_nodes = random.sample(merged_dataset, sample_size)
    weights = np.zeros((sample_size,))
    for i in range(0, len(end_nodes)):
        weights[i] = get_edge_value(start_node, end_nodes[i])
        if weights[i] < 0:
            weights[i] = 0
    weights = preprocessing.normalize(weights, norm='l1')
    weights = weights.reshape((sample_size,))
    ran = random.random()
    sample_value = 0
    current = 0
    while sample_value < ran and current < len(weights):
        sample_value = sample_value + weights[current]
        current = current + 1
    selected_node = end_nodes[current - 1]
    return selected_node


def generate_seqence(merged_dataset, sample_size, seq_length, start_node_set):
    # select start
    seq = list()
    start_node = random.sample(merged_dataset, 1)[0]
    (first, second, relation, _, _) = start_node
    start_node_str = first + '#' + second + '#' + relation
    while 1:
        if start_node_str in start_node_set:
            start_node = random.sample(merged_dataset, 1)[0]
            (first, second, relation, _, _) = start_node
            start_node_str = first + '#' + second + '#' + relation
        else:
            start_node_set.add(start_node_str)
            break
    seq.append(start_node)
    for i in range(0, seq_length):
        end_node = select_end_node(merged_dataset, sample_size, start_node)
        (first, second, relation, _, _) = end_node
        end_node_str = first + '#' + second + '#' + relation
        seq.append(end_node)
        start_node_set.add(end_node_str)
        start_node = end_node
    return seq, start_node_set


n_embeddings = 300
dataset_name = 'BLESS'
warnings.filterwarnings("ignore")

model_file = open('proj_' + dataset_name + '.pk', 'rb')
models = pickle.load(model_file)
model_file.close()

print('load fast text model...')
en_model = KeyedVectors.load_word2vec_format('fasttext.vec')
print('load fast text model success...')

relation_map, reverse_relation_map = generate_label_map()
relation_number = len(relation_map.keys())

training_set = load_training_set_with_dist(dataset_name + '/train.tsv')
testing_set = load_testing_set_with_dist(dataset_name + '/init_test.tsv')
merged_dataset = merge_datasets(training_set, testing_set)
print('dataset process success')

gamma = 2
file_count = 7
print(file_count)
start_node_set = set()

does_exists = os.path.exists(dataset_name + '_sample/')
if not does_exists:
    os.makedirs(dataset_name + '_sample/')

out_file = open(dataset_name + '_sample/' + str(file_count) + '.txt', 'w+')
for i in range(0, 1000):
    print('seq: ' + str(i))
    seq, start_node_set1 = generate_seqence(merged_dataset, sample_size=25, seq_length=50,
                                            start_node_set=start_node_set)
    for first, second, relation, dist, is_train in seq:
        out_file.write(first + '#' + second + ' ')
    out_file.write('\n')
    start_node_set = start_node_set1
out_file.close()
