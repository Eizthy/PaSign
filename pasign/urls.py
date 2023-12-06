from django.contrib import admin
from django.urls import path
from app.views import (LandingPageView, LoginPageView, DashboardView, ForgotPassView, TestView, UploadView, register_user, get_student_data)
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', LandingPageView.as_view(landing_page='index.html'), name='landing_page'),
    path('login/', LoginPageView.as_view(login_page='login-index.html'), name='login_page'),
    path('forgotpass/', ForgotPassView.as_view(), name='forgotpassword'),
    path('dashboard/', DashboardView.as_view(), name='dashboard_page'),
    path('upload/', UploadView.as_view(), name='upload_data'),
    path('register/', register_user, name='register_user'),
    path('test/', TestView.as_view(), name='test_data'),
    path('get-student-data/<str:student_id>/', get_student_data, name='get_student_data'),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
