from django.db import models


class JSONTransaction(models.Model):
    """
    JSON 처리 과정에서 로그 기록

    json_received_time: JSON을 받은 시간
    json_content: JSON 데이터
    response_received_time: 응답을 받은 시간
    response_data: 응답 데이터(감정값)
    """
    json_received_time = models.DateTimeField()
    json_content = models.TextField(blank=True, null=True)
    response_received_time = models.DateTimeField()
    response_data = models.IntegerField()

    def __str__(self):
        return f'Transaction at {self.json_received_time}'
