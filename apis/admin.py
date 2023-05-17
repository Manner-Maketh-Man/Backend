from django.contrib import admin
from .models import FileTransaction


class FileTransactionAdmin(admin.ModelAdmin):
    def file_received_time_formatted(self, obj):
        return obj.file_received_time.strftime('%Y-%m-%d %H:%M:%S')
    file_received_time_formatted.admin_order_field = 'file_received_time'
    file_received_time_formatted.short_description = 'File Received Time'

    def response_received_time_formatted(self, obj):
        return obj.response_received_time.strftime('%Y-%m-%d %H:%M:%S')
    response_received_time_formatted.admin_order_field = 'response_received_time'
    response_received_time_formatted.short_description = 'Response Received Time'

    list_display = ('file_received_time_formatted', 'response_received_time_formatted', 'response_data')
    change_list_template = 'apis/change_list.html'


admin.site.register(FileTransaction, FileTransactionAdmin)
