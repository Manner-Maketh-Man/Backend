from django.contrib import admin
from .models import FileTransaction


class FileTransactionAdmin(admin.ModelAdmin):
    list_display = ('file_received_time', 'response_received_time', 'response_data')
    change_list_template = 'Backend/change_list.html'


admin.site.register(FileTransaction, FileTransactionAdmin)
