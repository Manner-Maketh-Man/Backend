from django.contrib import admin
from .models import JSONTransaction


class JSONTransactionAdmin(admin.ModelAdmin):
    def json_received_time_formatted(self, obj):
        return obj.json_received_time.strftime('%Y-%m-%d %H:%M:%S')
    json_received_time_formatted.admin_order_field = 'json_received_time'
    json_received_time_formatted.short_description = 'Json Received Time'

    def response_received_time_formatted(self, obj):
        return obj.response_received_time.strftime('%Y-%m-%d %H:%M:%S')
    response_received_time_formatted.admin_order_field = 'response_received_time'
    response_received_time_formatted.short_description = 'Response Received Time'

    list_display = ('json_received_time_formatted', 'response_received_time_formatted', 'response_data')
    change_list_template = 'apis/change_list.html'


admin.site.register(JSONTransaction, JSONTransactionAdmin)
