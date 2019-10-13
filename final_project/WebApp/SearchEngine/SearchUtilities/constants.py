import logging
import sys
import os
import nltk
from pymystem3 import Mystem


ROOT_LOGGER = logging.getLogger()
ROOT_LOGGER.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
ROOT_LOGGER.addHandler(handler)

ROOT_LOGGER.info("Setting up projet environment...")

PROJECT_ROOT = "."
DB_NAME = "question_pairs.db"
logging.info("Make sure that you have internet connection to download data for vectorization.")
logging.info("Otherwise this program is to freeze!")
nltk.download('punkt')
MORPH = Mystem()
CORPUS_SIZE = 100000

FASTTEXT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w2v", "model.model")
FASTTEXT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w2v", "fast_text_matrix.npy")
ELMO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elmo", "elmo")