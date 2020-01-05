import pickle
import warnings
import numpy as np
from gensim.models import KeyedVectors


def load_dataset(path):
    dataset = set()
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
        dataset.add((first, second, relation))
    file.close()
    return dataset


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


def generate_feature_vector(dataset):
    F_matrix = np.zeros(shape=(len(dataset), n_embeddings * (relation_number + 0)))
    Y_matrix = np.zeros(shape=(len(dataset), 1))
    data_count = 0
    for first, second, relation in dataset:
        feature_vector = np.zeros(shape=(1, n_embeddings * (relation_number + 0)))
        for i in range(0, relation_number):
            relation_name = reverse_relation_map[i]
            W_matrix = models[relation_name]
            P_matrix = (
            np.dot(W_matrix, en_model[first].reshape(n_embeddings, 1)) - en_model[second].reshape(n_embeddings, 1)).T
            feature_vector[0, i * n_embeddings:(i + 1) * n_embeddings] = P_matrix
        F_matrix[data_count, :] = feature_vector
        relation_index = relation_map[relation]
        Y_matrix[data_count, :] = relation_index
        data_count = data_count + 1
    return F_matrix, Y_matrix


def generate_initial_prob(first, second, relation):
    temp_dataset = set()
    temp_dataset.add((first, second, relation))
    F_matrix_test, Y_matrix_test = generate_feature_vector(temp_dataset)
    Y_matrix_test_predict = lr.predict_proba(F_matrix_test)
    return Y_matrix_test_predict


n_embeddings = 300
dataset_name = 'BLESS'
warnings.filterwarnings("ignore")

model_file = open('proj_' + dataset_name + '.pk', 'rb')
models = pickle.load(model_file)
model_file.close()

lr_file = open('init_lr_' + dataset_name + '.pk', 'rb')
lr = pickle.load(lr_file)
lr_file.close()

print('load fast text model...')
en_model = KeyedVectors.load_word2vec_format('fasttext.vec')
print('load fast text model success...')

relation_map, reverse_relation_map = generate_label_map()
relation_number = len(relation_map.keys())

out_file = open(dataset_name + '/init_test.tsv', 'w+')
testing_set = load_dataset(dataset_name + '/test.tsv')
for first, second, relation in testing_set:
    out_str = first + '\t' + second + '\t' + relation
    probs = generate_initial_prob(first, second, relation)
    for i in range(0, relation_number):
        out_str = out_str + '\t' + str(probs[0, i])
    out_file.write(out_str + '\n')
out_file.close()
