from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from datetime import timedelta
from .models import FileTransaction


def handle_uploaded_file(file):
    # 파일이 실제로 존재하는지 확인
    if not file or file.size == 0:
        return None

    # TODO
    #   ML 모델로 파일 전송 구현
    #   ML 모델로부터 정수 데이터 수신 구현

    return 123  # 테스트로 123을 반환


@csrf_exempt
def process_file(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        file_received_time = timezone.now()
        response_data = handle_uploaded_file(file)

        # 파일이 존재하지 않는 경우, 에러 메시지를 반환합니다.
        if response_data is None:
            return JsonResponse({'error': 'No file or empty file'})

        response_received_time = timezone.now()

        file_transaction = FileTransaction.objects.create(
            file_received_time=file_received_time,
            response_received_time=response_received_time,
            response_data=response_data
        )
        file_transaction.save()

        # 데이터베이스에 저장된 데이터가 5개 이상인 경우, 가장 오래된 데이터를 삭제합니다.
        max_entries = 5
        if FileTransaction.objects.count() > max_entries:
            oldest_entry = FileTransaction.objects.order_by('file_received_time').first()
            oldest_entry.delete()

        # 24시간이 지난 데이터를 삭제합니다.
        time_threshold = timezone.now() - timedelta(days=1)
        expired_entries = FileTransaction.objects.filter(file_received_time__lt=time_threshold)
        expired_entries.delete()

        return JsonResponse({'response_data': response_data})

    return JsonResponse({'error': 'Invalid request method'})
