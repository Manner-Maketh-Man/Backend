from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from datetime import timedelta
from .models import JSONTransaction
from .run_models import handle_received_json


@csrf_exempt
def process_json(request):
    if request.method == 'POST':
        json_content = request.body.decode('utf-8')
        json_received_time = timezone.now()
        emotion_value = handle_received_json(json_content)

        # 데이터가 존재하지 않는 경우, 에러 메시지를 반환
        if emotion_value is None:
            return JsonResponse({'error': 'No data or wrong data'})

        response_received_time = timezone.now()

        json_transaction = JSONTransaction.objects.create(
            json_received_time=json_received_time,
            json_content=json_content,
            response_received_time=response_received_time,
            emotion_value=emotion_value,
        )
        json_transaction.save()

        # 데이터베이스에 저장된 데이터가 10개 이상인 경우, 가장 오래된 데이터를 삭제
        max_entries = 10
        if JSONTransaction.objects.count() > max_entries:
            oldest_entry = JSONTransaction.objects.order_by('json_received_time').first()
            oldest_entry.delete()

        # 24시간이 지난 데이터를 삭제
        time_threshold = timezone.now() - timedelta(days=1)
        expired_entries = JSONTransaction.objects.filter(json_received_time__lt=time_threshold)
        expired_entries.delete()

        # 정상적으로 처리된 경우, 응답 데이터를 반환
        return JsonResponse({'response_data': emotion_value})

    return JsonResponse({'error': 'Invalid request method'})
