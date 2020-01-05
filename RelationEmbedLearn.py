import warnings
import logging
from gensim.models import Word2Vec

n_embeddings = 300
dataset_name = 'BLESS'
warnings.filterwarnings("ignore")

count = 0
sentences = list()
for i in range(1, 6):
    in_file = open(dataset_name + '_sample/%d.txt' % i)
    while 1:
        line = in_file.readline()
        if not line:
            break
        line = line.replace('\n', '').strip()
        str = line.split(' ')
        sentences.append(str)
print('sentences load ok!')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = Word2Vec(sentences, min_count=1, window=5, sg=1)
model.wv.save_word2vec_format(dataset_name + '_r.model', binary=False)

