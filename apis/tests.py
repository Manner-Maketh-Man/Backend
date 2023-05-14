from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import FileTransaction
from django.utils import timezone


class FileTransactionModelTestCase(TestCase):
    def setUp(self):
        FileTransaction.objects.create(
            opposite_name='test_name',
            file_received_time=timezone.now(),
            response_received_time=timezone.now(),
            response_data=123
        )

    def test_file_transaction_creation(self):
        file_transaction = FileTransaction.objects.get(opposite_name='test_name')
        self.assertIsInstance(file_transaction, FileTransaction)
        self.assertEqual(file_transaction.response_data, 123)


class ProcessFileViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_process_file_view(self):
        file = SimpleUploadedFile("file.json", b"file_content", content_type="text/plain")
        response = self.client.post('/apis/process_file/', {'file': file}, format='multipart')

        # Test if the response status is 200 OK
        self.assertEqual(response.status_code, 200)

        # Test if the response data contains the opposite_name and recent_responses
        self.assertContains(response, 'opposite_name')
        self.assertContains(response, 'recent_responses')

    def test_process_file_view_no_file(self):
        response = self.client.post('/apis/process_file/', {}, format='multipart')

        # Test if the response status is 200 OK
        self.assertEqual(response.status_code, 200)

        # Test if the response data contains the error message
        self.assertContains(response, 'No file or empty file or No name')
