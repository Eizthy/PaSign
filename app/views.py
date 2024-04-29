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
        remember_me = request.POST.get('remember_me')  

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)

            if not remember_me:
                request.session.set_expiry(0)

            return redirect('dashboard_page')  
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

        print(f'signatures: {request.FILES}')

        if form.is_valid() and len(signatures) == 25:
            # Check for existing user with case-insensitive comparison
            student_id = form.cleaned_data['student_id']
            email = form.cleaned_data['email'].lower()

            # Check if a user with the same student ID or email already exists
            if UserData.objects.filter(student_id=student_id).exclude(id=form.instance.id).exists() or \
               UserData.objects.filter(email__iexact=email).exclude(id=form.instance.id).exists():
                messages.error(request, 'User with similar information already exists. Please check and try again.')
                return render(request, 'upload.html', {'form': form})

            user_data = form.save()

            signature_folder = get_user_signature_path(user_data, '')


            for i, signature_file in enumerate(signatures, start=1):
                file_extension = os.path.splitext(signature_file.name)[1]
                new_file_name = f'{student_id}_signature_{i}{file_extension}'
                new_file_path = os.path.join(settings.MEDIA_ROOT, signature_folder, new_file_name)
                
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

                with open(new_file_path, 'wb') as file:
                    file.write(signature_file.read())
                
                setattr(user_data, f'signature_{i}', os.path.join(signature_folder, new_file_name))

            user_data.save()

            # Add success message
            messages.success(request, 'Registration successful!')

            return redirect('upload_data')  # Redirect after successful registration

    else:
        form = UserDataForm()

    return render(request, 'upload.html', {'form': form})

#TEST
import base64

from time import perf_counter, sleep

import time


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

        st = time.time()
        start = perf_counter()

        if not UserData.objects.filter(student_id=student_id).exists():
            return render(request, self.test_data)

        user_data = UserData.objects.get(student_id=student_id) 
        bs64_img = convert_image_to_base64(img_file)

        result_1 = predict(img_file, user_data.signature_1)
        result_2 = predict(img_file, user_data.signature_2)
        result_3 = predict(img_file, user_data.signature_3)
        result_4 = predict(img_file, user_data.signature_4)
        result_5 = predict(img_file, user_data.signature_5)
        result_6 = predict(img_file, user_data.signature_6)
        result_7 = predict(img_file, user_data.signature_7)
        result_8 = predict(img_file, user_data.signature_8)
        result_9 = predict(img_file, user_data.signature_9)
        result_10 = predict(img_file, user_data.signature_10)
        result_11 = predict(img_file, user_data.signature_11)
        result_12 = predict(img_file, user_data.signature_12)
        result_13 = predict(img_file, user_data.signature_13)
        result_14 = predict(img_file, user_data.signature_14)
        result_15 = predict(img_file, user_data.signature_15)
        result_16 = predict(img_file, user_data.signature_16)
        result_17 = predict(img_file, user_data.signature_17)
        result_18 = predict(img_file, user_data.signature_18)
        result_19 = predict(img_file, user_data.signature_19)
        result_20 = predict(img_file, user_data.signature_20)
        result_21 = predict(img_file, user_data.signature_21)
        result_22 = predict(img_file, user_data.signature_22)
        result_23 = predict(img_file, user_data.signature_23)
        result_24 = predict(img_file, user_data.signature_24)
        result_25 = predict(img_file, user_data.signature_25)

        if result_1 <= 80:
            color_class_1 = "result-low"
        else:
            color_class_1 = "result-high"

        if result_2 <= 80:
            color_class_2 = "result-low"
        else:
            color_class_2 = "result-high"

        if result_3 <= 80:
            color_class_3 = "result-low"
        else:
            color_class_3 = "result-high"

        if result_4 <= 80:
            color_class_4 = "result-low"
        else:
            color_class_4 = "result-high"

        if result_5 <= 80:
            color_class_5 = "result-low"
        else:
            color_class_5 = "result-high"

        if result_6 <= 80:
            color_class_6 = "result-low"
        else:
            color_class_6 = "result-high"

        if result_7 <= 80:
            color_class_7 = "result-low"
        else:
            color_class_7 = "result-high"

        if result_8 <= 80:
            color_class_8 = "result-low"
        else:
            color_class_8 = "result-high"

        if result_9 <= 80:
            color_class_9 = "result-low"
        else:
            color_class_9 = "result-high"

        if result_10 <= 80:
            color_class_10 = "result-low"
        else:
            color_class_10 = "result-high"

        if result_11 <= 80:
            color_class_11 = "result-low"
        else:
            color_class_11 = "result-high"

        if result_12 <= 80:
            color_class_12 = "result-low"
        else:
            color_class_12 = "result-high"

        if result_13 <= 80:
            color_class_13 = "result-low"
        else:
            color_class_13 = "result-high"

        if result_14 <= 80:
            color_class_14 = "result-low"
        else:
            color_class_14 = "result-high"

        if result_15 <= 80:
            color_class_15 = "result-low"
        else:
            color_class_15 = "result-high"

        if result_16 <= 80:
            color_class_16 = "result-low"
        else:
            color_class_16 = "result-high"

        if result_17 <= 80:
            color_class_17 = "result-low"
        else:
            color_class_17 = "result-high"

        if result_18 <= 80:
            color_class_18 = "result-low"
        else:
            color_class_18 = "result-high"

        if result_19 <= 80:
            color_class_19 = "result-low"
        else:
            color_class_19 = "result-high"

        if result_20 <= 80:
            color_class_20 = "result-low"
        else:
            color_class_20 = "result-high"

        if result_21 <= 80:
            color_class_21 = "result-low"
        else:
            color_class_21 = "result-high"

        if result_22 <= 80:
            color_class_22 = "result-low"
        else:
            color_class_22 = "result-high"

        if result_23 <= 80:
            color_class_23 = "result-low"
        else:
            color_class_23 = "result-high"

        if result_24 <= 80:
            color_class_24 = "result-low"
        else:
            color_class_24 = "result-high"

        if result_25 <= 80:
            color_class_25 = "result-low"
        else:
            color_class_25 = "result-high"

        avg = (result_1 + result_2 + result_3 + result_4 + result_5 +
            result_6 + result_7 + result_8 + result_9 + result_10 +
            result_11 + result_12 + result_13 + result_14 + result_15 +
            result_16 + result_17 + result_18 + result_19 + result_20 +
            result_21 + result_22 + result_23 + result_24 + result_25) / 25

        if avg <= 80:
            avg_message = "The signature is considered as forged!"
            avg_class = "result-low"
        else:
            avg_message = "The signature is considered as genuine!"
            avg_class = "result-high"
            
        end = perf_counter()
        print(f"Time taken to execute code : {end-start}")
        
        # get the end time  
        et = time.time()

        # get the execution time
        elapsed_time = et - st
        
        print('Execution time:', elapsed_time, 'seconds')

        if elapsed_time - elapsed_time == 0:
            test_complete = "Test Complete"
            time_complete = f"Elapsed Time: {elapsed_time:.2f}s"
        

        context = {
            'result_1': f'{result_1:.2f}%',
            'result_2': f'{result_2:.2f}%',
            'result_3': f'{result_3:.2f}%',
            'result_4': f'{result_4:.2f}%',
            'result_5': f'{result_5:.2f}%',
            'result_6': f'{result_6:.2f}%',
            'result_7': f'{result_7:.2f}%',
            'result_8': f'{result_8:.2f}%',
            'result_9': f'{result_9:.2f}%',
            'result_10': f'{result_10:.2f}%',
            'result_11': f'{result_11:.2f}%',
            'result_12': f'{result_12:.2f}%',
            'result_13': f'{result_13:.2f}%',
            'result_14': f'{result_14:.2f}%',
            'result_15': f'{result_15:.2f}%',
            'result_16': f'{result_16:.2f}%',
            'result_17': f'{result_17:.2f}%',
            'result_18': f'{result_18:.2f}%',
            'result_19': f'{result_19:.2f}%',
            'result_20': f'{result_20:.2f}%',
            'result_21': f'{result_21:.2f}%',
            'result_22': f'{result_22:.2f}%',
            'result_23': f'{result_23:.2f}%',
            'result_24': f'{result_24:.2f}%',
            'result_25': f'{result_25:.2f}%',
            'avg_result': f'{avg:.2f}%',
            'avg_message': avg_message,
            'avg_class': avg_class,
            'color_class_1': color_class_1,
            'color_class_2': color_class_2,
            'color_class_3': color_class_3,
            'color_class_4': color_class_4,
            'color_class_5': color_class_5,
            'color_class_6': color_class_6,
            'color_class_7': color_class_7,
            'color_class_8': color_class_8,
            'color_class_9': color_class_9,
            'color_class_10': color_class_10,
            'color_class_11': color_class_11,
            'color_class_12': color_class_12,
            'color_class_13': color_class_13,
            'color_class_14': color_class_14,
            'color_class_15': color_class_15,
            'color_class_16': color_class_16,
            'color_class_17': color_class_17,
            'color_class_18': color_class_18,
            'color_class_19': color_class_19,
            'color_class_20': color_class_20,
            'color_class_21': color_class_21,
            'color_class_22': color_class_22,
            'color_class_23': color_class_23,
            'color_class_24': color_class_24,
            'color_class_25': color_class_25,
            'elapsed_time': elapsed_time,
            'signature_1': user_data.signature_1.url or '',
            'signature_2': user_data.signature_2.url or '',
            'signature_3': user_data.signature_3.url or '',
            'student_name': user_data.first_name + ' ' + user_data.last_name,
            'email': user_data.email or '',
            'student_id': student_id,
            'img': bs64_img,
            'test_complete': test_complete,
            'time_complete': time_complete,
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
            'signature_2_url': student.signature_2.url if student.signature_2 else '',
            'signature_3_url': student.signature_3.url if student.signature_3 else '',
        }
        return JsonResponse(data)
    except UserData.DoesNotExist:
        return JsonResponse({'error': 'Student not found'}, status=404)
