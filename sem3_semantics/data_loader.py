import os
import nltk
import pandas as pd
from tqdm import tqdm
from constants_loader import ROOT_LOGGER, MORPH, PROJECT_ROOT, CORPUS_SIZE
from project_exceptions import ReadingDataError


class DataSet:
    def __init__(self):
        self.__data = self.__read_table()
        self.questions = self.__get_questions()
        self.answers = self.__get_answer()
        self.target = self.__get_target()

    @staticmethod
    def __read_table():
        try:
            ROOT_LOGGER.info("Loading data...")
            for file in os.listdir(PROJECT_ROOT):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(PROJECT_ROOT, file)).dropna()
                    return df.loc[df["is_duplicate"] == 1]
        except Exception:
            raise ReadingDataError

    @staticmethod
    def __lemmatize(text):
        return [MORPH.lemmatize(token)[0] for token in nltk.word_tokenize(text)]

    def __get_questions(self):
        quest_arr = list(self.__data["question1"])[:CORPUS_SIZE]
        return [self.__lemmatize(quest_arr[item]) for item in tqdm(range(len(quest_arr)))]

    def __get_answer(self):
        ans_arr = list(self.__data["question2"])[:CORPUS_SIZE]
        return [self.__lemmatize(ans_arr[item]) for item in tqdm(range(len(ans_arr)))]

    def __get_target(self):
        return list(self.__data["is_duplicate"])[:CORPUS_SIZE]