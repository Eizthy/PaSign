from django.db import models
import os

def get_user_signature_path(instance, filename):
    """
        Creates a folder for each user using their student_id

        Example:
        >>> instance = UserData(student_id='1234567890')
        >>> get_user_signature_path(instance, 'signature_1.png')
        'Student 1234567890/signature_1.png'
        >>> get_user_signature_path(instance, 'signature_2.png')
        'Student 1234567890/signature_2.png'
        >>> get_user_signature_path(instance, 'signature_3.png')
        'Student 1234567890/signature_3.png'
    """
    folder_name = f"Student {instance.student_id}"
    return os.path.join(folder_name, filename)

class UserData(models.Model):
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    student_id = models.CharField(max_length=15)
    email = models.EmailField()
    signature_1 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_2 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_3 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_4 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_5 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_6 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_7 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_8 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_9 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_10 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_11 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_12 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_13 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_14 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_15 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_16 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_17 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_18 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_19 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_20 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_21 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_22 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_23 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_24 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)
    signature_25 = models.FileField(upload_to=get_user_signature_path, blank=True, null=True)

    def __str__(self):
        return f"{self.first_name} {self.last_name}"
