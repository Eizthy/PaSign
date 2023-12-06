import os
from django.views import View
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from .forms import UserDataForm
from django.http import JsonResponse
from .models import UserData, get_user_signature_path
from django.contrib import messages
from django.conf import settings
from .training import predict

# LANDING PAGE
class LandingPageView(View):
    landing_page = 'index.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.landing_page)

# LOGIN
class LoginPageView(View):
    login_page = 'landing/login-index.html'

    def get(self, request):
        return render(request, self.login_page)

    def post(self, request):
        username = request.POST.get('username')
        password = request.POST.get('password')
        remember_me = request.POST.get('remember_me')  # Check if the checkbox is checked

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)

            if not remember_me:
                # If "Remember Me" is not checked, set session expiry to 0 (browser session)
                request.session.set_expiry(0)

            return redirect('dashboard_page')  # Replace with the actual URL name for your dashboard page
        else:
            # Authentication failed, handle accordingly (e.g., show an error message)
            error_message = 'Invalid username or password. Please try again.'
            messages.error(request, error_message)
            return render(request, self.login_page, {'error_message': error_message})

        return render(request, self.login_page)
        
# FORGOT PASSWORD
class ForgotPassView(View):
    forgotpassword = 'forgotpass.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.forgotpassword)

# DASHBOARD
class DashboardView(View):
    dashboard_page = 'dashboard.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.dashboard_page)
    
# UPLOAD
class UploadView(View):
    upload_data = 'upload.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.upload_data)
    
# REGISTER STUDENT
def register_user(request):
    if request.method == 'POST':
        form = UserDataForm(request.POST, request.FILES)

        signatures = request.FILES.getlist('signature_files')

        if form.is_valid() and len(signatures) == 1:  # Adjust to handle one file
            # Check for existing user with case-insensitive comparison
            student_id = form.cleaned_data['student_id']
            email = form.cleaned_data['email'].lower()

            # Check if a user with the same student ID or email already exists
            if UserData.objects.filter(student_id=student_id).exclude(id=form.instance.id).exists() or \
               UserData.objects.filter(email__iexact=email).exclude(id=form.instance.id).exists():
                messages.error(request, 'User with similar information already exists. Please check and try again.')
                return render(request, 'upload.html', {'form': form})

            user_data = form.save(commit=False)  # Save the form but don't commit to the database yet

            # Use the get_user_signature_path function to determine the path
            signature_folder = get_user_signature_path(user_data, '')
            
            # Assuming only one file is selected
            signature_file = signatures[0]
            file_extension = os.path.splitext(signature_file.name)[1]
            new_file_name = f'{student_id}_signature{file_extension}'
            new_file_path = os.path.join(settings.MEDIA_ROOT, signature_folder, new_file_name)

            os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

            with open(new_file_path, 'wb') as file:
                file.write(signature_file.read())

            # Save the file path to the corresponding signature field in the model
            user_data.signature_1 = os.path.join(signature_folder, new_file_name)

            user_data.save()

            # Add success message
            messages.success(request, 'Registration successful!')

            return redirect('upload_data')  # Redirect after successful registration

    else:
        form = UserDataForm()

    return render(request, 'upload.html', {'form': form})

# TEST
import base64

def convert_image_to_base64(img_file):
    if img_file:
        image_content = img_file.read()
        base64_encoded = base64.b64encode(image_content).decode('utf-8')
        return base64_encoded
    return None

class TestView(View):
    test_data = 'test.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.test_data)

    def post(self, request, *args, **kwargs):
        is_verify_signature = request.POST.get('verify_signature')
        student_id = request.POST.get('student_id')
        img_file = request.FILES.get('img_file')

        if not UserData.objects.filter(student_id=student_id).exists():
            return render(request, self.test_data)

        user_data = UserData.objects.get(student_id=student_id) 
        bs64_img = convert_image_to_base64(img_file)

        result_1 = predict(img_file, user_data.signature_1)



        context = {
            'result_1': f'{result_1:.2f}%',
            'signature_1': user_data.signature_1.url or '',
            'student_name': user_data.first_name + ' ' + user_data.last_name,
            'email': user_data.email or '',
            'student_id': student_id,
            'img': bs64_img,
        }

        if is_verify_signature:
            return render(request, self.test_data, context)
        else:
            return render(request, self.test_data)

def get_student_data(request, student_id):
    try:
        student = UserData.objects.get(student_id=student_id)
        data = {
            'name': f"{student.first_name} {student.last_name}",
            'email': student.email,
            'signature_1_url': student.signature_1.url if student.signature_1 else '',
        }
        return JsonResponse(data)
    except UserData.DoesNotExist:
        return JsonResponse({'error': 'Student not found'}, status=404)
