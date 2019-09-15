from search_utils import FileProcess


if __name__ == '__main__':
    corpus = FileProcess()
    corpus.build_corpus()
    print("PRESS ENTER TO EXIT")
    request = input("~~~Enter your search request~~~\n")
    while request:
        response = corpus.search_in_file(request)
        for el in response:
            print(el)
        print("PRESS ENTER TO EXIT")
        request = input("~~~Would you like to search something else?~~~\n")
