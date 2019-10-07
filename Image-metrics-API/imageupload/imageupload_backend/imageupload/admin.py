from django.contrib import admin
from imageupload.models import UploadedImage

# Register the UploadedImage Model for the Admin Page
# admin.site.register(UploadedImage)

class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ('title','Nima_Score','noisescore','width','heigh','size')
admin.site.register(UploadedImage, UploadedImageAdmin)