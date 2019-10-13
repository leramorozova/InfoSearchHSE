import sqlite3
import os
import nltk
import pandas as pd
from WebApp.SearchEngine.SearchUtilities.project_utils import ProgressBar
from WebApp.SearchEngine.SearchUtilities.constants import ROOT_LOGGER, PROJECT_ROOT, DB_NAME, MORPH
from WebApp.SearchEngine.SearchUtilities.project_exceptions import ReadingDataError, DbInitError


class DbBuilder:
    def __init__(self):
        self.data_set = self.__read_data()
        self.conn = sqlite3.connect(DB_NAME)
        self.db = self.__init_database()

    @staticmethod
    def __read_data():
        try:
            corpus = []
            ROOT_LOGGER.info("Loading data...")
            for file in os.listdir(PROJECT_ROOT):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(PROJECT_ROOT, file)).dropna()
                    corpus = list(df["question1"]) + list(df["question2"])
            if corpus:
                for q in corpus:
                    yield q
        except Exception:
            ROOT_LOGGER.critical("Failed to read_data!")
            raise ReadingDataError

    def __init_database(self):
        db = self.conn.cursor()
        try:
            db.execute("DROP TABLE IF EXISTS sent_data")
            db.execute("""
                    CREATE TABLE sent_data
                    (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    sent TEXT NOT NULL,
                    lemmatized_sent TEXT NOT NULL);
                    """)
            ROOT_LOGGER.info("Database has been initialized successfully!")
            return db
        except Exception as e:
            raise DbInitError(e)

    def fill_database(self):
        ROOT_LOGGER.info("Filling the database...")
        pbar = ProgressBar(800000, fmt=ProgressBar.FULL)
        for i, row in enumerate(self.data_set):
            lemmatized_sent = " ".join([MORPH.lemmatize(token)[0] for token in nltk.word_tokenize(row)])
            self.db.execute('''
                            INSERT INTO sent_data
                            (sent, lemmatized_sent)
                            VALUES (?, ?);
                            ''', (row, lemmatized_sent))
            self.conn.commit()
            pbar.current += 1
            pbar()
        pbar.done()

    def close_db(self):
        self.conn.close()


if __name__ == "__main__":
    db_handler = DbBuilder()
    db_handler.fill_database()
    db_handler.close_db()