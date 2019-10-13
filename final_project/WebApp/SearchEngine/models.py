"""
        Данный файл содержит модели к каждой таблице базы данных проекта.
Классы и признаки названы в соответствии с названиями таблиц и столбцов базе данных.
Об устройсве БД читайте соответствующий раздел документации.
В случае необходорсти допустимо перемещать и переименовывать классы и переменные, но
    запрещено изменять содержимое подклассов Meta
"""
from django.db import models


class SentDataFields(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    sent = models.TextField()
    lemmatized_sent = models.TextField()

    class Meta:
        managed = False
        db_table = 'sent_data'
