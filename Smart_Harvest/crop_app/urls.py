from django.urls import path
from . import views

urlpatterns = [
    # Page URLs
    path('',           views.home,          name='home'),
    path('predict/',   views.predict,       name='predict'),
    path('history/',   views.history,       name='history'),
    path('about/',     views.about,         name='about'),

    # Auth URLs
    path('login/',     views.login_view,    name='login'),
    path('register/',  views.register_view, name='register'),
    path('logout/',    views.logout_view,   name='logout'),

    # Prediction API
    path('api/predict/',          views.api_predict,        name='api_predict'),
    path('api/history/',          views.api_history,        name='api_history'),
    path('api/history/<int:pk>/', views.api_history_detail, name='api_history_detail'),
    path('api/model-info/',       views.api_model_info,     name='api_model_info'),

    # User API
    path('api/users/',          views.api_users,        name='api_users'),
    path('api/users/<int:pk>/', views.api_user_detail,  name='api_user_detail'),
    path('api/register/',       views.api_register,     name='api_register'),
    path('api/login/',          views.api_login,        name='api_login'),
]