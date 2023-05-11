from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from datetime import timedelta
from .models import FileTransaction
import requests
from requests.exceptions import RequestException


def handle_uploaded_file(file):
    # 파일이 실제로 존재하는지 확인
    if not file or file.size == 0:
        return None

    # try:
    #     # 파일 객체를 이용하여 multipart/form-data 형태의 POST 요청을 보냄
    #     response = requests.post('http://ml_model_url/predict', files={'file': file}, timeout=5)  # Timeout : 5초
    #     # TODO
    #     #   ml_model_url을 실제 머신러닝 모델 URL로 변경해야 함
    # except RequestException as e:
    #     # 네트워크 관련 오류 발생 시
    #     print(f"Network error: {e}")
    #     return None
    #
    # # 서버로부터의 응답을 확인하고, 오류가 발생한 경우 None을 반환
    # if response.status_code != 200:
    #     print(f"Server error: {response.status_code}")
    #     return None
    #
    # try:
    #     # 응답에서 정수 데이터(감정값)를 가져옵니다.
    #     response_data = response.json()['prediction']
    #     # TODO
    #     #   실제 머신러닝 서버가 반환하는 응답의 형태에 따라 수정해야 함
    # except KeyError:
    #     # 응답 데이터에서 필요한 정보를 찾을 수 없는 경우
    #     print("Invalid response data")
    #     return None
    #
    # return response_data

    return 123  # 테스트로 123을 반환


@csrf_exempt
def process_file(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        name = request.POST.get('name')
        file_received_time = timezone.now()
        response_data = handle_uploaded_file(file)

        # 파일이 존재하지 않는 경우, 에러 메시지를 반환
        if response_data is None:
            return JsonResponse({'error': 'No file or empty file'})

        response_received_time = timezone.now()

        file_transaction = FileTransaction.objects.create(
            name=name,
            file_received_time=file_received_time,
            response_received_time=response_received_time,
            response_data=response_data
        )
        file_transaction.save()

        # 데이터베이스에 저장된 데이터가 5개 이상인 경우, 가장 오래된 데이터를 삭제
        max_entries = 5
        if FileTransaction.objects.count() > max_entries:
            oldest_entry = FileTransaction.objects.order_by('file_received_time').first()
            oldest_entry.delete()

        # 24시간이 지난 데이터를 삭제
        time_threshold = timezone.now() - timedelta(days=1)
        expired_entries = FileTransaction.objects.filter(file_received_time__lt=time_threshold)
        expired_entries.delete()

        # 정상적으로 처리된 경우, 응답 데이터를 반환
        return JsonResponse({'response_data': response_data})

    return JsonResponse({'error': 'Invalid request method'})
