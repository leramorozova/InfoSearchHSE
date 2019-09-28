import os
import logging
import sys
import time
import nltk
from copy import deepcopy
from pymystem3 import Mystem
from functools import reduce
import pandas as pd
from tqdm import tqdm
import numpy as np
from math import log
from sklearn.feature_extraction.text import CountVectorizer
from project_exceptions import ReadingDataError

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

logging.info("Setting up projet environment...")

PROJECT_ROOT = "."
CORPUS_SIZE = 38000
logging.info("Make sure that you have internet connection to download data for vectorization.")
logging.info("Otherwise this program is to freeze!")
nltk.download('punkt')
morph = Mystem()


class DataSet:
    def __init__(self):
        self.__data = self.__read_table()
        self.questions = self.__get_questions()
        self.answers = self.__get_answer()
        self.target = self.__get_target()

    @staticmethod
    def __read_table():
        try:
            logging.info("Loading data...")
            for file in os.listdir(PROJECT_ROOT):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(PROJECT_ROOT, file)).dropna()
                    return df.loc[df["is_duplicate"] == 1]
        except:
            raise ReadingDataError

    @staticmethod
    def __lemmatize(text):
        return " ".join([morph.lemmatize(token)[0] for token in nltk.word_tokenize(text)])

    def __get_questions(self):
        quest_arr = list(self.__data["question1"])[:CORPUS_SIZE]
        return [self.__lemmatize(quest_arr[item]) for item in tqdm(range(len(quest_arr)))]

    def __get_answer(self):
        ans_arr = list(self.__data["question2"])[:CORPUS_SIZE]
        return [self.__lemmatize(ans_arr[item]) for item in tqdm(range(len(ans_arr)))]

    def __get_target(self):
        return list(self.__data["is_duplicate"])[:CORPUS_SIZE]


class BuildingTfIDF:
    """
    За документ буду считать каждое вхождение вопроса в датасете.
    Надеюсь, я правильно думаю.
    """
    def __init__(self):
        self.corpus = DataSet()
        self.doc_lens, self.av_len = self.__get_av_len()
        self.count_vec = CountVectorizer(input="content", ngram_range=(1, 1))
        self.doc_count = len(self.corpus.answers)
        self.__tf_matrix = self.__vectorize_data()
        self.bm32_tf_matrix = self.__get_bm25_tf()
        self.__vocabulary = self.__get_vocabulary()
        self.__word_indexes = self.__get_word_indexes()
        self.idf = self.__get_idfs()

    def __get_av_len(self):
        lens = [len(doc) for doc in self.corpus.answers]
        return lens, sum(lens) / len(lens)

    def __vectorize_data(self):
        logging.info("Vectorizing data...")
        return self.count_vec.fit_transform(self.corpus.answers).toarray() / np.array(self.doc_lens).reshape((-1, 1))

    def __get_vocabulary(self):
        return self.count_vec.get_feature_names()

    def __get_word_indexes(self):
        return {word: i for i, word in enumerate(self.count_vec.get_feature_names())}

    def __count_idf(self, word):
        doc_with_token = self.count_docs_with_term(word)
        return log((self.doc_count - doc_with_token + 0.5) / (doc_with_token + 0.5), 2)

    def __get_idfs(self):
        logging.info("\nCounting idfs...")
        return np.array([self.__count_idf(self.__vocabulary[term]) for term in tqdm(range(len(self.__vocabulary)))])

    def count_docs_with_term(self, term):
        idx = self.__word_indexes[term] if term in self.__word_indexes else -1
        count = 0
        for doc in self.__tf_matrix:
            count = + 1 if idx > -1 and doc[idx] else + 0
        return count

    def __get_bm25_tf(self, b=0.75, k=2):
        logging.info("Building TF-IDF matrix...")
        vectors = deepcopy(self.__tf_matrix)
        for i in range(self.__tf_matrix.shape[0]):
            vectors[i] = (self.__tf_matrix[i] * (k + 1.0)) / \
                         (self.__tf_matrix[i] + k * (1.0 - b + b * (self.doc_lens[i] / self.av_len)))
        return vectors

    def get_freq_in_doc(self, doc_idx, term):
        return self.__tf_matrix[doc_idx][self.__word_indexes[term]] if term in self.__word_indexes else 0

    def get_doc_length(self, doc_idx):
        return len(self.__tf_matrix[doc_idx])

    def get_doc_text(self, doc_idx):
        return self.corpus.answers[doc_idx]


class BM25(BuildingTfIDF):
    @staticmethod
    def get_max_doc(vector):
        return set(reduce(lambda x, y: x + y,
                      [np.argwhere(vector == top).flatten().tolist() for top in np.sort(vector)[-5:]]))

    @staticmethod
    def compute_metric(responses):
        return np.sum(responses) / len(responses)

    def bm25_iter(self, query, b=0.75, k=2):
        result_vec = np.zeros([self.doc_count])
        query_proc = CountVectorizer(input="content", ngram_range=(1, 1))
        _ = query_proc.fit_transform([query])
        query_wordlist = query_proc.get_feature_names()
        for word in query_wordlist:
            doc_with_token = self.count_docs_with_term(word)
            idf = log((self.doc_count - doc_with_token + 0.5) / (doc_with_token + 0.5), 2)
            for doc in range(self.doc_count):
                tf = self.get_freq_in_doc(doc, word)
                doc_len = self.get_doc_length(doc)
                result_vec[doc] += idf * (tf * (k + 1) / (tf + k * (1 - b + b * (doc_len / self.av_len))))
        return result_vec

    def bm25_matrix(self, query, b=0.75, k=2):
        query_vec = self.count_vec.transform([query]).toarray()
        idf_vector = self.idf * query_vec
        return self.bm32_tf_matrix.dot(idf_vector.reshape(idf_vector.shape[1],))

    def is_true_answer_in_response(self, question_id, response):
        return question_id in self.get_max_doc(response)

    def generate_responce(self, bm_func, b=0.75, k=2, task_text="", metrics=False, query=None):
        start_time = time.time()
        logging.info("\n" + task_text)
        if query:
            query = " ".join([morph.lemmatize(token)[0] for token in nltk.word_tokenize(query)])
            responce = self.get_max_doc(bm_func(query, b, k))
            if len(responce) == self.doc_count:
                logging.info("Nothing was found :(")
            else:
                logging.info("Documents with max metrics:")
                logging.info(responce)
        else:
            if metrics:
                result = self.compute_metric([self.is_true_answer_in_response(q, bm_func(self.corpus.questions[q], b, k))
                                             for q in tqdm(range(self.doc_count))])
                logging.info(f"\nResult metric: {result}")
        logging.info(f"\nRequest processing takes {time.time() - start_time}\n\n")


if __name__ == "__main__":
    bm25 = BM25()
    logging.info("\nCompare iterative and matrix search")
    bm25.generate_responce(bm25.bm25_iter, task_text="Testing iterative version")
    bm25.generate_responce(bm25.bm25_matrix, task_text="Testing matrix version")
    bm25.generate_responce(bm25.bm25_iter, b=0.75, task_text="BM25", metrics=True)
    bm25.generate_responce(bm25.bm25_iter, b=0, task_text="BM15", metrics=True)
    bm25.generate_responce(bm25.bm25_iter, b=1, task_text="BM11", metrics=True)
    bm25.generate_responce(bm25.bm25_iter, b=0.75, task_text="Search X-mas holiday", query="рождественские каникулы")