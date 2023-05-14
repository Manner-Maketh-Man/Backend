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
        return None, None

    # try:
    #     # 파일 객체를 이용하여 multipart/form-data 형태의 POST 요청을 보냄
    #     response = requests.post('http://ml_model_url/predict', files={'file': file}, timeout=5)
    #     # TODO
    #     #   ml_model_url을 실제 머신러닝 모델 URL로 변경해야 함
    # except RequestException as e:
    #     # 네트워크 관련 오류 발생 시
    #     print(f"Network error: {e}")
    #     return None, None
    #
    # # 서버로부터의 응답을 확인하고, 오류가 발생한 경우 None을 반환
    # if response.status_code != 200:
    #     print(f"Server error: {response.status_code}")
    #     return None, None
    #
    # try:
    #     # TODO
    #     #   실제 머신러닝 서버가 반환하는 응답의 형태에 따라 수정해야 함
    #     # 응답에서 이름 데이터를 가져옴
    #     opposite_name = response.json()['opposite_name']
    #     # 응답에서 정수 데이터(감정값)를 가져옴
    #     response_data = response.json()['prediction']
    # except KeyError:
    #     # 응답 데이터에서 필요한 정보를 찾을 수 없는 경우
    #     print("Invalid response data")
    #     return None, None
    #
    # return opposite_name, response_data

    return 'test_name', 123  # 테스트용


@csrf_exempt
def process_file(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        file_received_time = timezone.now()
        opposite_name, response_data = handle_uploaded_file(file)

        # 파일이 존재하지 않는 경우, 에러 메시지를 반환
        if response_data is None or opposite_name is None:
            return JsonResponse({'error': 'No file or empty file or No name'})

        response_received_time = timezone.now()

        file_transaction = FileTransaction.objects.create(
            file_received_time=file_received_time,
            response_received_time=response_received_time,
            opposite_name=opposite_name,
            response_data=response_data
        )
        file_transaction.save()

        # 데이터베이스에 저장된 데이터가 10개 이상인 경우, 가장 오래된 데이터를 삭제
        max_entries = 10
        if FileTransaction.objects.count() > max_entries:
            oldest_entry = FileTransaction.objects.order_by('file_received_time').first()
            oldest_entry.delete()

        # 24시간이 지난 데이터를 삭제
        time_threshold = timezone.now() - timedelta(days=1)
        expired_entries = FileTransaction.objects.filter(file_received_time__lt=time_threshold)
        expired_entries.delete()

        # 정상적으로 처리된 경우, 응답 데이터를 반환
        recent_responses_data = get_recent_responses(opposite_name)
        return JsonResponse({'opposite_name': opposite_name, 'recent_responses': recent_responses_data})

    return JsonResponse({'error': 'Invalid request method'})


def get_recent_responses(opposite_name, num_entries=5):  # 최근 응답 데이터 중 최대 5개를 가져옴
    entries = FileTransaction.objects.filter(opposite_name=opposite_name).order_by('-response_received_time')[:num_entries]
    return [entry.response_data for entry in entries]
