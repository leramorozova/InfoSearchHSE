from search_engine import WordToVecSearch, Bm25Search, ElmoSearch
from constants_loader import ROOT_LOGGER

if __name__ == "__main__":
    result = []

    wv = WordToVecSearch()
    result.append(wv.search_quality)

    elmo = ElmoSearch()
    result.append(elmo.search_quality)

    lm = Bm25Search()
    result.append(lm.search_quality)

    ROOT_LOGGER.info("====RESULT STATICTICS=====")
    for el in result:
        ROOT_LOGGER.info(el)