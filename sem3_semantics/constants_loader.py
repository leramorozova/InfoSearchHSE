import logging
import os
import sys
import nltk
from pymystem3 import Mystem

ROOT_LOGGER = logging.getLogger()
ROOT_LOGGER.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
ROOT_LOGGER.addHandler(handler)

ROOT_LOGGER.info("Setting up projet environment...")

PROJECT_ROOT = "."
W2V_MODEL = os.path.join(PROJECT_ROOT, "w2v", "model.model")
ELMO_MODEL = os.path.join(PROJECT_ROOT, "elmo", "elmo")
CORPUS_SIZE = 300
logging.info("Make sure that you have internet connection to download data for vectorization.")
logging.info("Otherwise this program is to freeze!")
nltk.download('punkt')
MORPH = Mystem()
