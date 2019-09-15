import os
import zipfile
import nltk
import re
import logging
import sys
from pymystem3 import Mystem
from project_exceptions import NoZipfile

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

logging.info("Setting up search service...")
PROJECT_ROOT = '.'
nltk.download('punkt')
mystem_analyzer = Mystem()


class FileItem:
    def __init__(self, text, file_name, idx):
        self.text = text
        self.file_name = file_name
        self.id = idx
        self.tokens = nltk.word_tokenize(self.text)
        self.parsed_text = self.clean_text()

    def clean_text(self):
        lemmas = [mystem_analyzer.lemmatize(word)[0] for word in self.tokens]
        clean_lemmas = [re.sub("\n", "", word) for word in lemmas if word != "\ufeff"]
        return clean_lemmas


class FileProcess:
    def __init__(self):
        logging.info("Reading archive...")
        self.working_archive = 0
        self.files = []
        self.word_dict = {}
        for file in os.listdir(PROJECT_ROOT):
            if zipfile.is_zipfile(file):
                self.working_archive = os.path.join(PROJECT_ROOT, file)
        if not self.working_archive:
            raise NoZipfile(PROJECT_ROOT)

    def __read_files(self):
        working_directory = os.path.join(PROJECT_ROOT, self.working_archive)
        zf = zipfile.ZipFile(working_directory, "r")
        logging.info("File processing started.")
        for idx, name in enumerate(zf.namelist()):
            text = zf.read(name).decode()
            file_item = FileItem(text, name, idx)
            self.files.append(file_item)

    def __build_corpus(self):
        self.__read_files()
        logging.info("Building indexes...")
        for file in self.files:
            for word in file.parsed_text:
                if word not in self.word_dict:
                    self.word_dict[word] = [file.id]
                else:
                    if file.id not in self.word_dict[word]:
                        self.word_dict[word].append(file.id)

    @staticmethod
    def __process_search_params(text):
        tokens = nltk.word_tokenize(text)
        lemmas = [mystem_analyzer.lemmatize(word)[0] for word in tokens]
        return lemmas

    def search_in_file(self, search_param):
        self.__build_corpus()
        found_words = []
        for word in self.__process_search_params(search_param):
            if word in self.word_dict:
                found_words.append(set(self.word_dict[word]))
        get_files = []
        for arr in found_words:
            if not get_files:
                get_files = arr
            else:
                get_files = get_files & arr
        return get_files


if __name__ == '__main__':
    corpus = FileProcess()
    # TODO: искать сначала слова рядом
    print(corpus.search_in_file("я тебя люблю"))