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


dataset_name = 'BLESS'
training_set = load_dataset(dataset_name + '/train.tsv')
relation_set = load_relation_set(training_set)
output_file = open('rel_map_' + dataset_name + '.txt', 'w+')
i = 0
for relation in relation_set:
    output_file.write(relation + '\t%d\n' % i)
    i = i + 1
output_file.close()
