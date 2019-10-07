import os
import uuid
import numpy
import numpy as np
import subprocess
import cv2
import json 
from skimage.restoration import estimate_sigma
from PIL import Image
from django.db import models
from django.conf import settings
import hurry
import re
from hurry.filesize import size , alternative
from .utils import utils
import keras
import tensorflow as tf
import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

#---------------------------------- TENSORFLOW SERVING FUNCTIONS ------------------------------------------------

TFS_HOST = 'localhost'
TFS_PORT = 8500


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def get_image_quality_predictions(image_path, model_name):
    # Load and preprocess image
    image = utils.load_image(image_path, target_size=(224, 224))
    image = keras.applications.mobilenet.preprocess_input(image)

    # Run through model
    target = f'{TFS_HOST}:{TFS_PORT}'
    channel = grpc.insecure_channel(target)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'image_quality'

    request.inputs['input_image'].CopyFrom(
        tf.contrib.util.make_tensor_proto(np.expand_dims(image, 0))
    )

    response = stub.Predict(request, 10.0)
    result = round(calc_mean_score(response.outputs['quality_prediction'].float_val), 2)

    return json.dumps({'mean_score_prediction': np.round(result, 3)}, indent=2)
#---------------------------------------------- END -------------------------------------------------------------


def scramble_uploaded_filename(instance, filename):
    """
    Scramble / uglify the filename of the uploaded file, but keep the files extension (e.g., .jpg or .png)
    :param instance:
    :param filename:
    :return:
    """
    extension = filename.split(".")[-1]
    return "{}.{}".format(uuid.uuid4(), extension)

def create_thumbnail(input_image, thumbnail_size=(256, 256)):
    """
    Create a thumbnail of an existing image
    :param input_image:
    :param thumbnail_size:
    :return:
    """
    # make sure an image has been set
    if not input_image or input_image == "":
        return

    # open image
    image = Image.open(input_image)

    # use PILs thumbnail method; use anti aliasing to make the scaled picture look good
    image.thumbnail(thumbnail_size, Image.ANTIALIAS)

    # parse the filename and scramble it
    filename = scramble_uploaded_filename(None, os.path.basename(input_image.name))
    arrdata = filename.split(".")
    # extension is in the last element, pop it
    extension = arrdata.pop()
    basename = "".join(arrdata)
    # add _thumb to the filename
    new_filename = basename + "_thumb." + extension
    new_filename = re.sub(r"[^.,a-zA-Z0-9]+", '', new_filename)
    # save the image in MEDIA_ROOT and return the filename
    image.save(os.path.join(settings.MEDIA_ROOT, new_filename))

    return new_filename

def get_noise(input_image):
    image = Image.open(input_image)
    open_cv_image = numpy.array(image) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    value = estimate_sigma(open_cv_image, multichannel=True, average_sigmas=True)
    return (value)

def get_dimentions(input_image):
    image = Image.open(input_image)
    open_cv_image = numpy.array(image) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    
    height, width, depth = open_cv_image.shape
    return (height,width)

def save_orignal(input_image):
    # make sure an image has been set
    if not input_image or input_image == "":
        return

    # open image
    image = Image.open(input_image)
    image = image.convert("RGB")
    if str(input_image.name).find('.png') != -1:
        savename = input_image.name.split('.')
        savename = re.sub(r"[^.,a-zA-Z0-9]+", '', savename[0])
        image.save(os.path.join(settings.MEDIA_ROOT, savename+'.jpg'),quality=95)
        input_image.name = savename+'.jpg'
        print("Save Orignal png : "+savename+'.jpg')
    else:
        image.save(os.path.join(settings.MEDIA_ROOT, input_image.name),quality=95)
        print("Save Orignal not png : "+input_image.name)
    raster_size = os.path.getsize(os.path.join(settings.MEDIA_ROOT, input_image.name))
    raster_size = size(raster_size, system=alternative)
    metrics = raster_size.split(' ')
    if metrics[1] == 'KB':
        metrics[0] =  int(metrics[0])/1000
    # os.remove(os.path.join(settings.MEDIA_ROOT, inp
    # 
    # ut_image.name))
    return round(metrics[0],3)
    
class UploadedImage(models.Model):
    """
    Provides a Model which contains an uploaded image aswell as a thumbnail
    """
    image = models.ImageField("Uploaded image", upload_to=scramble_uploaded_filename)

    # thumbnail
    thumbnail = models.ImageField("Thumbnail of uploaded image", blank=True)

    # title and description
    title = models.CharField("Title of the uploaded image", max_length=255, default="Unknown Picture")
    description = models.TextField("Description of the uploaded image", default="")
    size = models.DecimalField(null=True,decimal_places=3,max_digits=6) 
    width = models.IntegerField(null=True) 
    heigh = models.IntegerField(null=True)
    noisescore = models.FloatField(null=True)
    Nima_Score = models.FloatField(null=True)
    Status = models.TextField("Image Quality Assisment",default="No Assisment")

    def __str__(self):
        return self.title

    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        """
        On save, generate a new thumbnail
        :param force_insert:
        :param force_update:
        :param using:
        :param update_fields:
        :return:
        """
        # Get sigma value of the uploaded image
        #self.image_metrics = image_metrics(self.image)
        # generate and set thumbnail or none
        self.thumbnail = create_thumbnail(self.image)
        self.noisescore = round(numpy.float32(get_noise(self.image)),2)
        self.width = get_dimentions(self.image)[1]
        self.heigh = get_dimentions(self.image)[0]
        self.size = save_orignal(self.image)
        if str(self.image.name).find('.png') != -1:
            savename = self.image.name.split('.')
            self.image.name = savename+".jpg"
            print("SECOND CHECK PNG : "+
                  self.image.name)
        else:
            print("SECOND CHECK NOT PNG : "+self.image.name)
        print("AFTER CONVERSIONS THE NAME IS : ", self.image.name)
        # bashCommand = "/Users/harishkhollam/Documents/image-quality-assessment/predict --docker-image nima-cpu --base-model-name MobileNet --weights-file /Users/harishkhollam/Documents/image-quality-assessment/models/MobileNet/weights_mobilenet_technical_0.11.hdf5 --image-source /Users/harishkhollam/Documents/Image-metrics-API/django-rest-imageupload-example/uploaded_media/"+self.image.name+" | grep mean_score_prediction"
        # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        # output, error = process.communicate()
        # context  = str(output).split('mean_score_prediction')
        # self.Nima_Score = round(numpy.float32(re.sub(r"[^.0-9]+", '', context[1])),4)
        test = get_image_quality_predictions('/Users/harishkhollam/Documents/Image-metrics-API/django-rest-imageupload-example/uploaded_media/'+self.image.name,'mobilenet_technical')
        print("The test output is \n",round(numpy.float32(re.sub(r"[^.0-9]+", '', test)),4))
        self.Nima_Score = round(numpy.float32(re.sub(r"[^.0-9]+", '', test)),4)
        if self.Nima_Score >= 3.98 and self.noisescore <= 0.8:
            self.Status = "Good"
        else:
            self.Status = "Bad"
        # os.remove(os.path.join(settings.MEDIA_ROOT, self.image.name))
        # self.size = get_size(self.image)
        # Check if a pk has been set, meaning that we are not creating a new image, but updateing an existing one
        #if self.pk:
        #    force_update = True

        # force update as we just changed something
        super(UploadedImage, self).save(force_update=force_update)

