from gensim.models import KeyedVectors
import numpy as np
import warnings
import pickle


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


def load_relation_set(dataset):
    relation_set = set()
    for first, second, relation in dataset:
        relation_set.add(relation)
    return relation_set


def load_sub_dataset(dataset, relation_name):
    sub_dataset = set()
    for first, second, relation in dataset:
        if relation == relation_name:
            sub_dataset.add((first, second))
    return sub_dataset


def train_matrix(sub_dataset):
    X_matrix = np.zeros(shape=(len(sub_dataset), n_embeddings))
    Z_matrix = np.zeros(shape=(len(sub_dataset), n_embeddings))
    i = 0
    for first, second in sub_dataset:
        X_matrix[i] = en_model[first]
        Z_matrix[i] = en_model[second]
        i = i + 1
    W_matrix = np.dot(np.linalg.inv(np.dot(X_matrix.T, X_matrix) + para_lambda * np.eye(n_embeddings)),
                      np.dot(X_matrix.T, Z_matrix))
    return W_matrix


def train_all_matrices(dataset):
    models = dict()
    relation_set = load_relation_set(dataset)
    for relation in relation_set:
        sub_dataset = load_sub_dataset(dataset, relation)
        W_matrix = train_matrix(sub_dataset)
        models[relation] = W_matrix
    return models


para_lambda = 0.001
n_embeddings = 300
dataset_name = 'BLESS'
warnings.filterwarnings("ignore")
print('load fast text model...')
en_model = KeyedVectors.load_word2vec_format('fasttext.vec')
print('load fast text model success...')

dataset = load_dataset(dataset_name + '/train.tsv')
models = train_all_matrices(dataset)
model_file = open('proj_' + dataset_name + '.pk', 'wb')
pickle.dump(models, model_file)
model_file.close()
