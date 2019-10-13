from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('search_result', views.search_result, name='search_result'),
    path('failed_result', views.search_result, name='failed_result')
]
