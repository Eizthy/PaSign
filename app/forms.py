from django import forms
from .models import UserData

class UserDataForm(forms.ModelForm):
    class Meta:
        model = UserData
        fields = ['first_name', 'last_name', 'student_id', 'email']
