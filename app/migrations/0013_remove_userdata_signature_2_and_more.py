# Generated by Django 4.2.7 on 2023-12-06 14:24

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0012_rename_signature_files_userdata_signature_1_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userdata',
            name='signature_2',
        ),
        migrations.RemoveField(
            model_name='userdata',
            name='signature_3',
        ),
    ]
