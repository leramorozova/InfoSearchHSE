import nltk
from scipy.spatial.distance import cosine
from .constants import FASTTEXT_MODEL, FASTTEXT_CACHE, ROOT_LOGGER, CORPUS_SIZE, MORPH, ELMO_MODEL
from .project_exceptions import ModelLoaderError
from .project_utils import ResponseItem
from .db_loader import Database
import numpy as np
import os
from gensim.models import KeyedVectors
from copy import deepcopy
from math import log
from tqdm import tqdm
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .elmo.elmo_helpers import get_elmo_vectors, load_elmo_embeddings


class DefaultSearch:
    def __init__(self):
        pass

    @staticmethod
    def load_data():
        db = Database()
        result = db.execute(f"""SELECT lemmatized_sent FROM sent_data
                                LIMIT {CORPUS_SIZE}""", 0)
        db.close()
        return [el[0] for el in result]

    def __load_model(self):
        pass

    @staticmethod
    def load_matrix(model_root):
        return np.load(model_root)

    def fit(self):
        pass

    def transform(self, doc):
        pass

    def search(self, query):
        pass


class WordToVecSearch(DefaultSearch):
    def __init__(self):
        super(WordToVecSearch, self).__init__()
        self.__model = self.__load_model()
        #if not os.path.isfile(FASTTEXT_CACHE):
        self.__data = self.load_data()
        self.__corpus_matrix = self.fit() #if not os.path.isfile(FASTTEXT_CACHE) else self.load_matrix(FASTTEXT_CACHE)

    def __load_model(self):
        ROOT_LOGGER.info("Initializing W2V model...")
        try:
            return KeyedVectors.load(FASTTEXT_MODEL)
        except Exception:
            ROOT_LOGGER.critical("Initialization failed!")
            raise ModelLoaderError(FASTTEXT_MODEL)

    def transform(self, doc):
        lemmas_vectors = np.zeros((len(doc), self.__model.vector_size))
        vec = np.zeros((self.__model.vector_size,))

        for idx, lemma in enumerate(doc):
            if lemma in self.__model.wv:
                lemmas_vectors[idx] = self.__model.wv[lemma]

        if lemmas_vectors.shape[0] is not 0:
            vec = np.mean(lemmas_vectors, axis=0)
        return vec

    def fit(self):
        ROOT_LOGGER.info("Building W2V matrix...")
        matrix = np.zeros((CORPUS_SIZE, self.__model.vector_size))
        for i in tqdm(range(len(self.__data))):
            doc_vec = self.transform(self.__data[i].split())
            matrix[i] = doc_vec
        ROOT_LOGGER.info("Built.")
        np.save(FASTTEXT_CACHE, matrix)
        return matrix

    def search(self, query):
        ROOT_LOGGER.info("Searching...")
        response = []
        query_vec = self.transform([MORPH.lemmatize(token)[0] for token in nltk.word_tokenize(query)])
        for i, row in enumerate(self.__corpus_matrix):
            response.append(ResponseItem(i, cosine(row, query_vec)))
        return sorted(response, reverse=True)


class ElmoSearch(DefaultSearch):
    def __init__(self):
        super(ElmoSearch, self).__init__()
        self.__time = 0
        self.__data = self.load_data()
        self.__batcher, self.__ids, self.__sent_input = self.__load_model()
        self.__corpus_matrix = self.__build_up_matrix()

    def __load_model(self):
        ROOT_LOGGER.info("Loading ELMO model...")
        tf.reset_default_graph()
        return load_elmo_embeddings(ELMO_MODEL)

    def transform(self, doc):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            vec = get_elmo_vectors(sess, [doc], self.__batcher, self.__ids, self.__sent_input)[0]
        return vec

    def __build_up_matrix(self):
        ROOT_LOGGER.info("Building ELMO matrix...")
        full_vectors = []
        for i in tqdm(range(len(self.__data))):
            full_vectors.append(self.transform(self.__data[i]))
        ROOT_LOGGER.info("Built.")
        return full_vectors

    def search(self, query):
        ROOT_LOGGER.info("Searching...")
        response = []
        proc_q = " ".join([MORPH.lemmatize(token)[0] for token in nltk.word_tokenize(query)])
        query_vec = self.transform(proc_q)
        for i, row in enumerate(self.__corpus_matrix):
            response.append(ResponseItem(i, np.sum(cosine_similarity(row, query_vec)) / len(proc_q)))
        return sorted(response, reverse=True)


class TfIdfSearch(DefaultSearch):
    def __init__(self):
        super(TfIdfSearch, self).__init__()
        self.__data = self.load_data()
        self.count_vec = CountVectorizer(input="content", ngram_range=(1, 1))
        self.doc_count = len(self.__data)
        self.__tf_idf = self.fit()

    def fit(self):
        ROOT_LOGGER.info("Vectorizing data...")
        transformer = TfidfTransformer()
        return transformer.fit_transform(self.count_vec.fit_transform(self.__data).toarray())

    def transform(self, doc):
        return self.count_vec.transform([doc])

    def search(self, query):
        ROOT_LOGGER.info("Search...")
        response = []
        query_vec = self.count_vec.transform([" ".join([MORPH.lemmatize(token)[0] for
                                                        token in nltk.word_tokenize(query)])]).toarray().reshape(-1, 1)
        vector = self.__tf_idf * query_vec
        for i, value in enumerate(vector):
            if value[0] != 0:
                response.append(ResponseItem(i, value[0]))
        return sorted(response)


class Bm25Search(DefaultSearch):
    def __init__(self):
        super(Bm25Search, self).__init__()
        self.__data = self.load_data()
        self.doc_lens, self.av_len = self.__get_av_len()
        self.count_vec = CountVectorizer(input="content", ngram_range=(1, 1))
        self.doc_count = len(self.__data)
        self.__tf_matrix = self.fit()
        self.bm32_tf_matrix = self.__get_bm25_tf()
        self.__vocabulary = self.__get_vocabulary()
        self.__word_indexes = self.__get_word_indexes()
        self.idf = self.__get_idfs()

    def __get_av_len(self):
        lens = [len(doc) for doc in self.__data]
        return lens, sum(lens) / len(lens)

    def fit(self):
        ROOT_LOGGER.info("Vectorizing data...")
        return self.count_vec.fit_transform(self.__data).toarray() / np.array(self.doc_lens).reshape((-1, 1))

    def __get_vocabulary(self):
        return self.count_vec.get_feature_names()

    def __get_word_indexes(self):
        return {word: i for i, word in enumerate(self.count_vec.get_feature_names())}

    def __count_idf(self, word):
        doc_with_token = self.count_docs_with_term(word)
        return log((self.doc_count - doc_with_token + 0.5) / (doc_with_token + 0.5), 2)

    def __get_idfs(self):
        ROOT_LOGGER.info("\nCounting idfs...")
        return np.array([self.__count_idf(self.__vocabulary[term]) for term in tqdm(range(len(self.__vocabulary)))])

    def count_docs_with_term(self, term):
        idx = self.__word_indexes[term] if term in self.__word_indexes else -1
        count = 0
        for doc in self.__tf_matrix:
            count = + 1 if idx > -1 and doc[idx] else + 0
        return count

    def __get_bm25_tf(self, b=0.75, k=2):
        ROOT_LOGGER.info("Building TF-IDF matrix...")
        vectors = deepcopy(self.__tf_matrix)
        for i in range(self.__tf_matrix.shape[0]):
            vectors[i] = (self.__tf_matrix[i] * (k + 1.0)) / \
                         (self.__tf_matrix[i] + k * (1.0 - b + b * (self.doc_lens[i] / self.av_len)))
        return vectors

    def get_freq_in_doc(self, doc_idx, term):
        return self.__tf_matrix[doc_idx][self.__word_indexes[term]] if term in self.__word_indexes else 0

    def get_doc_length(self, doc_idx):
        return len(self.__tf_matrix[doc_idx])

    def search(self, query):
        ROOT_LOGGER.info("Search...")
        response = []
        query_vec = self.count_vec.transform([" ".join([MORPH.lemmatize(token)[0] for
                                                        token in nltk.word_tokenize(query)])]).toarray()
        idf_vector = self.idf * query_vec
        result_vec = self.bm32_tf_matrix.dot(idf_vector.reshape(-1, 1))
        for i, value in enumerate(result_vec):
            if value[0] != 0:
                response.append(ResponseItem(i, value[0]))
        return sorted(response)