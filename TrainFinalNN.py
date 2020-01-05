import pickle
import warnings
import numpy as np
from gensim.models import KeyedVectors
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


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
    F_matrix = np.zeros(shape=(len(dataset), n_embeddings * (relation_number+3) + n_relation_embeddings))
    Y_matrix = np.zeros(shape=(len(dataset), 1))
    data_count = 0
    for first, second, relation in dataset:
        feature_vector = np.zeros(shape=(1, n_embeddings * (relation_number+3)))
        relation_notation = first + '#' + second
        relation_embeddings = np.zeros(shape=(n_relation_embeddings,))
        if relation_notation in relation_model:
            relation_embeddings = relation_model[relation_notation]
        else:
            relation_embeddings = infer_relation_embeddings(first, second)
        #    print("not found")
        for i in range(0, relation_number):
            relation_name = reverse_relation_map[i]
            W_matrix = models[relation_name]
            P_matrix = (
                np.dot(W_matrix, en_model[first].reshape(n_embeddings, 1)) - en_model[second].reshape(n_embeddings,
                                                                                                      1)).T
            feature_vector[0, i * n_embeddings:(i + 1) * n_embeddings] = P_matrix
        feature_vector[0, relation_number * n_embeddings:(relation_number + 1) * n_embeddings] = en_model[first]
        feature_vector[0, (relation_number+1) * n_embeddings:(relation_number + 2) * n_embeddings] = en_model[second]
        feature_vector[0, (relation_number+2) * n_embeddings:(relation_number + 3) * n_embeddings] = en_model[first]-en_model[second]

        relation_embeddings = relation_embeddings.reshape((1, n_relation_embeddings))
        feature_vector = np.hstack((feature_vector, relation_embeddings))
        F_matrix[data_count, :] = feature_vector
        relation_index = relation_map[relation]
        Y_matrix[data_count, :] = relation_index
        data_count = data_count + 1
    return F_matrix, Y_matrix


def infer_relation_embeddings(first, second):
    first_sim = en_model.similar_by_word(first, topn=20)
    second_sim = en_model.similar_by_word(second, topn=20)
    count = 0
    current_embeddings = np.zeros(shape=(1,n_relation_embeddings))
    for word1, _ in first_sim:
        for word2, _ in second_sim:
            relation_notation = word1 + '#' + word2
            if relation_notation in relation_model:
                current_embeddings = current_embeddings + relation_model[relation_notation]
                count = count + 1
    if count > 0:
        current_embeddings = current_embeddings / count
        #print('inferred')
    else:
        current_embeddings = np.random.normal(0, 0.1, n_relation_embeddings)
    return current_embeddings


n_embeddings = 300
n_relation_embeddings = 100
dataset_name = 'BLESS'
warnings.filterwarnings("ignore")

model_file = open('proj_' + dataset_name + '.pk', 'rb')
models = pickle.load(model_file)
model_file.close()

print('load fast text model...')
en_model = KeyedVectors.load_word2vec_format('fasttext.vec')
print('load fast text model success...')

print('load relation model...')
relation_model = KeyedVectors.load_word2vec_format(dataset_name + '_r.model')
print('load relation model success...')

relation_map, reverse_relation_map = generate_label_map()
relation_number = len(relation_map.keys())

training_set = load_dataset(dataset_name + '/train.tsv')
F_matrix_train, Y_matrix_train = generate_feature_vector(training_set)

cls = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(300,), early_stopping=True, verbose=True)
cls.fit(F_matrix_train, Y_matrix_train)

testing_set = load_dataset(dataset_name + '/test.tsv')
F_matrix_test, Y_matrix_test = generate_feature_vector(testing_set)

Y_matrix_test_predict = cls.predict(F_matrix_test)
print(classification_report(Y_matrix_test, Y_matrix_test_predict, digits=3))
