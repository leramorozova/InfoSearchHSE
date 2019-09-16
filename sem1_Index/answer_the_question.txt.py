from search_utils import FileProcess
from collections import Counter


def most_common_word(corpus):
    """
    какое слово является самым частотным?
    ALARM: вывожу 20 слов, потому что не удаляла стоп-слова
    """
    count = Counter([])
    for f in corpus.files:
        new_count = Counter(f.parsed_text)
        count.update(new_count)
    print("===САМЫЕ ЧАСТОТНЫЕ СЛОВА===")
    for el in count.most_common(20):
        print(el[0].lemma, "-", el[1], "раз")
    print("\n")


def least_common_word(corpus):
    """
    какое самым редким
    """
    count = Counter([])
    for f in corpus.files:
        new_count = Counter(f.parsed_text)
        count.update(new_count)
    print("===НАИМЕНЕЕ ЧАСТОТНЫЕ СЛОВА===")
    for el in count.most_common()[:-20-1:-1]:
        print(el[0].lemma, "-", el[1], "раз")
    print("\n")


def all_doc_words(corpus):
    """
    какой набор слов есть во всех документах коллекции
    """
    all_docs = []
    for word in corpus.lemma_dict:
        if len(corpus.lemma_dict[word]) == len(corpus.files) - 7:
            all_docs.append(word.lemma)
    print("СЛОВА, КОТОРЫЕ ЕСТЬ ВО ВСЕХ ДОКУМЕНТАХ: " + " ".join(all_docs) + "\n")


def most_popular_season(corpus):
    """
    какой сезон был самым популярным у Чендлера? у Моники?
    """
    chen_count = []
    chen_response = corpus.search_in_file("Чендлер")
    for r in chen_response:
        chen_count.append(r.filename[0])
    monica_count = []
    monica_response = corpus.search_in_file("Моника")
    for r in monica_response:
        monica_count.append(r.filename[0])
    chen_count = Counter(chen_count)
    monica_count = Counter(monica_count)
    print("CAMЫЙ ПОПУЛЯРНЫЙ СЕЗОН У ЧЕНДЕРА:", chen_count.most_common(1)[0][0])
    print("CAMЫЙ ПОПУЛЯРНЫЙ СЕЗОН У МОНИКИ:", monica_count.most_common(1)[0][0])


if __name__ == '__main__':
    corpus = FileProcess()
    corpus.build_corpus()
    most_common_word(corpus)
    least_common_word(corpus)
    all_doc_words(corpus)
    most_popular_season(corpus)

