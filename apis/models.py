from django.db import models


class FileTransaction(models.Model):
    """
    파일 처리 과정에서의 로그 저장

    file_received_time: 파일을 받은 시간
    response_received_time: 응답을 받은 시간
    response_data: 응답 데이터(감정값)
    """
    file_received_time = models.DateTimeField()
    response_received_time = models.DateTimeField()
    response_data = models.IntegerField()

    def __str__(self):
        return f'Transaction at {self.file_received_time}'
