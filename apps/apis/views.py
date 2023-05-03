from django.http import JsonResponse
from django.utils import timezone
from .models import FileTransaction


def handle_uploaded_file(file):
    # 다른 서버로 파일을 전송하고 정수 데이터를 받아옵니다.
    # 여기서는 예시로 123을 반환하도록 합니다.
    return 123


def process_file(request):
    if request.method == 'POST':
        file = request.FILES['file']
        file_received_time = timezone.now()
        response_data = handle_uploaded_file(file)
        response_received_time = timezone.now()

        file_transaction = FileTransaction.objects.create(
            file_received_time=file_received_time,
            response_received_time=response_received_time,
            response_data=response_data
        )
        file_transaction.save()

        return JsonResponse({'response_data': response_data})

    return JsonResponse({'error': 'Invalid request method'})
