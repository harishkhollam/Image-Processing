# -*- coding: utf-8 -*-
# Generated by Django 1.10.8 on 2019-10-04 05:05
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imageupload', '0004_uploadedimage_size'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uploadedimage',
            name='size',
            field=models.IntegerField(null=True),
        ),
    ]
