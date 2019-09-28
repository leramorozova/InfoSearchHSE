class ReadingDataError(Exception):
    def __init__(self):
        msg = f"Dataset was not pushed or pulled correclty. Please, try to get it by " \
              f"yourself: https://www.kaggle.com/loopdigga/quora-question-pairs-russian"
        super().__init__(msg)