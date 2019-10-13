import re
import sys
from .db_loader import Database


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


class ResponseItem:
    def __init__(self, doc_idx, metric):
        self.__doc_idx = doc_idx + 1
        self.metric = metric
        self.doc_text = self.__find_doc()

    def __lt__(self, other):
        return self.metric > other.metric

    def __repr__(self):
        return f"ResponseItem({self.metric, self.doc_text})"

    def __find_doc(self):
        db = Database()
        result = db.execute(f"""SELECT sent FROM sent_data
                                WHERE id=%s""", (self.__doc_idx,))
        db.close()
        return result[0][0]
