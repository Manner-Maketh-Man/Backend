from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, Client
from django.utils import timezone
from .models import FileTransaction
import json


class FileTransactionModelTest(TestCase):
    def test_create_file_transaction(self):
        file_received_time = timezone.now()
        response_received_time = timezone.now()
        response_data = 123

        file_transaction = FileTransaction.objects.create(
            file_received_time=file_received_time,
            response_received_time=response_received_time,
            response_data=response_data
        )

        self.assertEqual(file_received_time, file_transaction.file_received_time)
        self.assertEqual(response_received_time, file_transaction.response_received_time)
        self.assertEqual(response_data, file_transaction.response_data)


class ProcessFileViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_process_file_view(self):
        test_json_data = {'key': 'value'}
        test_json_file = SimpleUploadedFile("test_json_file.json", json.dumps(test_json_data).encode(), content_type="application/json")
        response = self.client.post('/apis/process_file/', {'file': test_json_file})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['response_data'], 123)  # handle_uploaded_file에서 반환된 값
