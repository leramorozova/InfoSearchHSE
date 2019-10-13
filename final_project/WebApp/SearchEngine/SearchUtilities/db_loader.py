from django.db import connection


class Database:
    def __init__(self):
        self.cur = connection.cursor()

    def commit(self):
        self.cur.commit()

    def execute(self, q, arg):
        if arg != 0:
            self.cur.execute(q, arg)
        else:
            self.cur.execute(q)
        res = self.cur.fetchall()
        return res

    def get_data(self, data, if_none):
        try:
            str_data = data[0][0]
        except IndexError:
            if if_none == 1:
                str_data = None
            else:
                str_data = "нет данных"
        return str_data

    def close(self):
        self.cur.close()
