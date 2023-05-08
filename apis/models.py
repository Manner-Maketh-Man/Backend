# models.py
from django.db import models


class FileTransaction(models.Model):
    file_received_time = models.DateTimeField()
    response_received_time = models.DateTimeField()
    response_data = models.IntegerField()

    def __str__(self):
        return f'Transaction at {self.file_received_time}'
