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


class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
                          r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


class TokenItem:
    def __init__(self, token):
        self.token = token
        self.lemma = mystem_analyzer.lemmatize(token)[0]

    def __eq__(self, other):
        return self.lemma == other.lemma

    def __hash__(self):
        return hash(self.lemma)


class FileItem:
    def __init__(self, text, file_name, idx):
        self.text = text
        self.file_name = file_name
        self.id = idx
        self.tokens = nltk.word_tokenize(self.text)
        self.parsed_text = self.__clean_text()

    def __len__(self):
        return len(self.tokens)

    def __clean_text(self):
        return [TokenItem(word) for word in self.tokens]


class ResponseItem:
    def __init__(self, context, file):
        self.filename = self.prettify_name(file.file_name)
        self.context = [" ".join(sub_context) for sub_context in context]

    @staticmethod
    def prettify_name(filename):
        split_name = filename.split(" - ")
        season = split_name[-2]
        series = split_name[-1].split(".")[0]
        ret_filename = season + "\t" + series
        return ret_filename

    def __str__(self):
        return f"====={self.filename}=====\n" + '\n'.join(self.context) + "\n\n"


class FileProcess:
    def __init__(self):
        logging.info("Reading archive...")
        self.working_archive = 0
        self.files = []
        self.lemma_dict = {}
        for file in os.listdir(PROJECT_ROOT):
            if zipfile.is_zipfile(file):
                self.working_archive = os.path.join(PROJECT_ROOT, file)
        if not self.working_archive:
            raise NoZipfile(PROJECT_ROOT)

    def __read_files(self):
        working_directory = os.path.join(PROJECT_ROOT, self.working_archive)
        zf = zipfile.ZipFile(working_directory, "r")
        logging.info("File processing started. It will get some time.")
        progress = ProgressBar(len(zf.namelist()), fmt=ProgressBar.FULL)
        for idx, name in enumerate(zf.namelist()):
            if "Friends" in name:
                text = zf.read(name).decode()
                file_item = FileItem(text, name, idx)
                self.files.append(file_item)
            progress.current += 1
            progress()
        progress.done()

    def build_corpus(self):
        self.__read_files()
        logging.info("\nBuilding indexes...")
        for file in self.files:
            for word in file.parsed_text:
                if word not in self.lemma_dict:
                    self.lemma_dict[word] = [file.id]
                else:
                    if file.id not in self.lemma_dict[word]:
                        self.lemma_dict[word].append(file.id)

    @staticmethod
    def __process_search_params(text):
        tokens = nltk.word_tokenize(text)
        lemmas = [TokenItem(word) for word in tokens]
        return lemmas

    def __get_file_with_id(self, idx):
        for file in self.files:
            if file.id == idx:
                return file

    @staticmethod
    def __get_context(idx, file):
        start = 0 if idx < 5 else idx - 5
        if idx == 0:
            left_context = []
        else:
            left_context = ['...'] + [file.tokens[i] for i in range(start, idx)]
        if len(file) - 1 == idx:
            right_context = []
        else:
            end = len(file) if len(file) < idx + 5 else idx + 5
            right_context = [file.tokens[i] for i in range(idx + 1, end)] + ['...']
        point_word = ['\033[1m' + file.tokens[idx] + '\033[0m']
        return left_context + point_word + right_context

    @staticmethod
    def __get_words_from_text(text, req):
        idxs = []
        for word in req:
            for i, token in enumerate(text):
                if token == word:
                    idxs.append(i)
        return idxs

    def __build_up_contexts(self, r, req):
        contexts = []
        for idx in r:
            file = self.__get_file_with_id(idx)
            token_indexes = self.__get_words_from_text(file.parsed_text, req)
            sub_context = []
            for item in token_indexes:
                sub_context.append(self.__get_context(item, file))
            contexts.append(ResponseItem(sub_context, file))
        return contexts

    def search_in_file(self, search_param):
        logging.info("Performing search")
        found_words = []
        request = self.__process_search_params(search_param)
        for word in request:
            if word in self.lemma_dict:
                found_words.append(set(self.lemma_dict[word]))
        if len(found_words) != len(request):
            logging.warning("\nUnfortunately, some words of your request cannot be found in the corpus.\n")
        get_files = []
        for arr in found_words:
            if not get_files:
                get_files = arr
            else:
                get_files = get_files & arr
        return self.__build_up_contexts(get_files, request)