from django.test import TestCase, Client
from django.utils import timezone
from .models import JSONTransaction
from .views import get_recent_responses
import json


class JSONTransactionTestCase(TestCase):
    def setUp(self):
        # Initialize a client object
        self.client = Client()

    def test_process_json_post(self):
        # Prepare data to post
        data_to_post = {
            "key1": "value1",
            "key2": "value2"
        }
        response = self.client.post('/apis/process_json/',
                                    data=json.dumps(data_to_post),
                                    content_type='application/json')

        # Check the status code of the response
        self.assertEqual(response.status_code, 200)

        # Check the response data
        response_data = response.json()
        self.assertIn('recent_responses', response_data)

        # Check if the transaction is recorded in the database
        transaction = JSONTransaction.objects.first()
        self.assertIsNotNone(transaction)
        self.assertEqual(json.loads(transaction.json_content), data_to_post)
        self.assertEqual(transaction.emotion_value, 123)

    def test_process_json_invalid_request_method(self):
        # Send a GET request
        response = self.client.get('/apis/process_json/')
        self.assertEqual(response.status_code, 200)

        # Check the response data
        response_data = response.json()
        self.assertEqual(response_data, {'error': 'Invalid request method'})

    def test_get_recent_responses(self):
        # Create 10 JSONTransaction instances
        for i in range(10):
            JSONTransaction.objects.create(
                json_received_time=timezone.now(),
                json_content=json.dumps({"key": "value"}),
                response_received_time=timezone.now(),
                response_data=i
            )

        # Call the function
        recent_responses = get_recent_responses()

        # Check the length of the response
        self.assertEqual(len(recent_responses), 5)

        # Check the content of the response
        self.assertEqual(recent_responses, [9, 8, 7, 6, 5])
