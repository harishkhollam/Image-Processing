from rest_framework import serializers
from imageupload.models import UploadedImage


# Serializer for UploadedImage Model
# serializes pk and image
class UploadedImageSerializer(serializers.ModelSerializer):
    """
    Serializer for the UPloadedImage Model
    Provides the pk, image, thumbnail, title and description
    """
    class Meta:
        model = UploadedImage
        fields = ('pk', 'image','thumbnail','title', 'description','size','width','heigh','noisescore','Nima_Score','Status' )
        read_only_fields = ('thumbnail',)

