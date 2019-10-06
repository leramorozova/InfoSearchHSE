import numpy as np
from gensim.models import KeyedVectors
from time import time
from functools import reduce
from tqdm import tqdm
from copy import deepcopy
from math import log
import tensorflow as tf
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from constants_loader import W2V_MODEL, ROOT_LOGGER, CORPUS_SIZE, ELMO_MODEL
from project_exceptions import ModelLoaderError
from data_loader import DataSet
from elmo.elmo_helpers import get_elmo_vectors, load_elmo_embeddings


class DefaultSearch:
    def __init__(self):
        pass

    @classmethod
    def get_max_doc(cls, vector):
        return set(reduce(lambda x, y: x + y,
                   [np.argwhere(vector == top).flatten().tolist() for top in np.sort(vector)[:5]]))


class WordToVecSearch(DefaultSearch):
    def __init__(self):
        super(WordToVecSearch, self).__init__()
        self._time = 0
        self.__model = self.__load_model()
        self.__data = DataSet()
        self.__corpus_matrix = self.__build_up_matrix()

    @staticmethod
    def __load_model():
        ROOT_LOGGER.info("Initializing W2V model...")
        try:
            return KeyedVectors.load(W2V_MODEL)
        except Exception:
            ROOT_LOGGER.critical("Initialization failed!")
            raise ModelLoaderError(W2V_MODEL)

    def __vectorize_doc(self, doc):
        lemmas_vectors = np.zeros((len(doc), self.__model.vector_size))
        vec = np.zeros((self.__model.vector_size,))

        for idx, lemma in enumerate(doc):
            if lemma in self.__model.wv:
                lemmas_vectors[idx] = self.__model.wv[lemma]

        if lemmas_vectors.shape[0] is not 0:
            vec = np.mean(lemmas_vectors, axis=0)
        return vec

    def __build_up_matrix(self):
        ROOT_LOGGER.info("Building W2V matrix...")
        start = time()
        matrix = np.zeros((CORPUS_SIZE, self.__model.vector_size))
        for i, doc in enumerate(self.__data.answers):
            doc_vec = self.__vectorize_doc(doc)
            matrix[i] = doc_vec
        self.__time = time() - start
        ROOT_LOGGER.info("Built.")
        return matrix

    @property
    def search_quality(self):
        ROOT_LOGGER.info("Computing W2V metrics...")
        metrics = []
        for i in tqdm(range(len(self.__data.questions))):
            responce_vec = []
            query_vec = self.__vectorize_doc(self.__data.questions[i])
            for row in self.__corpus_matrix:
                responce_vec.append(cosine(row, query_vec))
            metrics += [1] if i in self.get_max_doc(responce_vec) else [0]
        return f"W2V - Index time: {self.__time}\nSearch_metrics: {np.sum(metrics) / len(metrics)}"


class ElmoSearch(DefaultSearch):
    def __init__(self):
        super(ElmoSearch, self).__init__()
        self.__time = 0
        self.__data = DataSet()
        self.__batcher, self.__ids, self.__sent_input = self.__load_model()
        self.__corpus_matrix = self.__build_up_matrix()

    @staticmethod
    def __load_model():
        ROOT_LOGGER.info("Loading ELMO model...")
        tf.reset_default_graph()
        return load_elmo_embeddings(ELMO_MODEL)

    def __vectorize_doc(self, doc):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            vec = get_elmo_vectors(sess, [doc], self.__batcher, self.__ids, self.__sent_input)[0]
        return vec

    def __build_up_matrix(self):
        ROOT_LOGGER.info("Building ELMO matrix...")
        start = time()
        full_vectors = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(200, CORPUS_SIZE + 1, 200):
                sentences = self.__data.answers[i-200:i]
                elmo_vec = get_elmo_vectors(sess, sentences, self.__batcher, self.__ids, self.__sent_input)
                for vec in elmo_vec:
                    full_vectors.append(vec)
        ROOT_LOGGER.info("Built.")
        self.__time = time() - start
        return full_vectors

    @property
    def search_quality(self):
        ROOT_LOGGER.info("Computing ELMO metrics...")
        metrics = []
        for i in tqdm(range(len(self.__data.questions))):
            responce_vec = []
            query_vec = self.__vectorize_doc(self.__data.questions[i])
            for row in self.__corpus_matrix:
                responce_vec.append(cosine_similarity(row, query_vec, dense_output=False))
            metrics += [1] if i in self.get_max_doc(responce_vec) else [0]
        return f"ELMO - Index time: {self.__time}\nSearch_metrics: {np.sum(metrics) / len(metrics)}"


class Bm25Search(DefaultSearch):
    """
    За документ буду считать каждое вхождение вопроса в датасете.
    Надеюсь, я правильно думаю.
    """
    def __init__(self):
        super(Bm25Search, self).__init__()
        self.__data = [" ".join(ans) for ans in DataSet().answers]
        self.__quest = [" ".join(q) for q in DataSet().questions]
        self.__time = time()
        self.doc_lens, self.av_len = self.__get_av_len()
        self.count_vec = CountVectorizer(input="content", ngram_range=(1, 1))
        self.doc_count = len(self.__data)
        self.__tf_matrix = self.__vectorize_data()
        self.bm32_tf_matrix = self.__get_bm25_tf()
        self.__vocabulary = self.__get_vocabulary()
        self.__word_indexes = self.__get_word_indexes()
        self.idf = self.__get_idfs()

    def __get_av_len(self):
        lens = [len(doc) for doc in self.__data]
        return lens, sum(lens) / len(lens)

    def __vectorize_data(self):
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

    @property
    def search_quality(self):
        ROOT_LOGGER.info("Computing ELMO metrics...")
        metrics = []
        for i in tqdm(range(len(self.__quest))):
            query_vec = self.count_vec.transform([self.__quest[i]]).toarray()
            idf_vector = self.idf * query_vec
            metrics += [1] if i in self.get_max_doc(self.bm32_tf_matrix.dot(idf_vector.reshape(idf_vector.shape[1],)) *
                                                    -1) else [0]
        return f"BM25 - Index time: {time() - self.__time}\nSearch_metrics: {np.sum(metrics) / len(metrics)}"
