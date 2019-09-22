import os
import logging
import sys
import time
import nltk
from pymystem3 import Mystem
from functools import reduce
import pandas as pd
from tqdm import tqdm
import numpy as np
from math import log
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

PROJECT_ROOT = "."
nltk.download('punkt')
morph = Mystem()


class DataSet:
    """
    Беру только 30 тыщ пар вопросов и ответов, потому что на большее CountVectorizer
    на моем железе не способен
    """
    def __init__(self):
        self.__data = self.__read_table()
        self.questions = self.__get_questions()
        self.answers = self.__get_answer()
        self.target = self.__get_target()

    @staticmethod
    def __read_table():
        logging.info("Loading data...")
        for file in os.listdir(PROJECT_ROOT):
            if file.endswith(".csv"):
                return pd.read_csv(os.path.join(PROJECT_ROOT, file)).dropna()

    @staticmethod
    def __lemmatize(text):
        return " ".join([morph.lemmatize(token)[0] for token in nltk.word_tokenize(text)])

    def __get_questions(self):
        return list(self.__data["question1"])

    def __get_answer(self):
        ans_arr = list(self.__data["question2"])[:35000]
        return [self.__lemmatize(ans_arr[item]) for item in tqdm(range(len(ans_arr)))]

    def __get_target(self):
        return list(self.__data["is_duplicate"])


class BuildingTfIDF:
    """
    За документ буду считать каждое вхождение вопроса в датасете.
    Надеюсь, я правильно думаю.
    """
    def __init__(self):
        self.corpus = DataSet()
        self.av_len = self.__get_av_len()
        self.count_vec = CountVectorizer(input="content", ngram_range=(1, 1))
        self.doc_count = len(self.corpus.answers)
        self.tf_idf = TfidfTransformer()
        self.__tf_matrix = self.__vectorize_data()
        self.tf_idf_matrix = self.__get_tfidf()
        self.__word_indexes = self.__get_word_indexes()

    def __get_av_len(self):
        lens = [len(doc) for doc in self.corpus.answers]
        return sum(lens) / len(lens)

    def __vectorize_data(self):
        logging.info("Vectorizing data...")
        res = self.count_vec.fit_transform(self.corpus.answers).toarray()
        np.save("count_vect.py", res)
        return res

    def __get_word_indexes(self):
        return {word: i for i, word in enumerate(self.count_vec.get_feature_names())}

    def __get_tfidf(self):
        logging.info("Building TF-IDF matrix...")
        tf_idf_matrix = self.tf_idf.fit_transform(self.__tf_matrix)
        return tf_idf_matrix

    def count_docs_with_term(self, term):
        idx = self.__word_indexes[term] if term in self.__word_indexes else -1
        count = 0
        for doc in self.__tf_matrix:
            count = + 1 if idx > -1 and doc[idx] else + 0
        return count

    def get_freq_in_doc(self, doc_idx, term):
        return self.__tf_matrix[doc_idx][self.__word_indexes[term]] if term in self.__word_indexes else 0

    def get_doc_length(self, doc_idx):
        return len(self.__tf_matrix[doc_idx])

    def get_doc_text(self, doc_idx):
        return self.corpus.answers[doc_idx]


class BM25(BuildingTfIDF):
    @staticmethod
    def __get_max_doc(vector):
        return reduce(lambda x, y: x + y,
                     [np.argwhere(vector == top).flatten().tolist() for top in np.sort(vector)[-5:]])

    @staticmethod
    def compute_metric(responses):
        return np.sum(responses) / len(responses)

    def __count_idf(self, word):
        doc_with_token = self.count_docs_with_term(word)
        return log((self.doc_count - doc_with_token + 0.5) / (doc_with_token + 0.5), 2)

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
        query_vec = query_vec.reshape((query_vec.shape[1], query_vec.shape[0]))
        res = self.tf_idf_matrix.dot(query_vec)
        return res

    def is_true_answer_in_response(self, question_id, response):
        return question_id in self.__get_max_doc(response)

    def generate_responce(self, bm_func, b=0.75, k=2, task_text=""):
        logging.info("\n" + task_text)
        start_time = time.time()
        result = self.compute_metric([self.is_true_answer_in_response(q, bm_func(self.corpus.questions[q], b, k))
                                     for q in tqdm(range(self.doc_count))])
        logging.info(f"\nRequest processing takes {time.time() - start_time}")
        logging.info(f"Result metric: {result}\n\n")


if __name__ == "__main__":
    bm25 = BM25()
    # logging.info("Compare iterative and matrix search")
    # bm25.generate_responce(bm25.bm25_iter, task_text="Testing iterative version")
    # bm25.generate_responce(bm25.bm25_matrix, task_text="testing iterative version")
    # bm25.generate_responce(bm25.bm25_iter, b=0.75, task_text="BM25")
    # bm25.generate_responce(bm25.bm25_iter, b=0, task_text="BM15")
    # bm25.generate_responce(bm25.bm25_iter, b=1, task_text="BM11")