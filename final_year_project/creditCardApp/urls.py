from django.urls import path
from django.contrib import admin
from .import views





urlpatterns = [

    # path('admin/', admin.site.urls),
    # path('home/',views.home),
    #
    # path('home/about/',views.about),
    # path('home/login/', views.login),
    # path('index/', views.index, name="index"),
    # path('index/upload/',views.uploadFile,name="upload"),
    #
    # path('index/logout/',views.logout)

    path('admin/', admin.site.urls),
    path('home/', views.home,name='home'),

    path('home/about/', views.about),
    path('home/login/', views.loginPage,name='login'),
    path('home/register/', views.register),
    path('index/', views.index, name='index'),
    path('index/upload/', views.uploadFile, name="upload"),
    path('index/report/', views.showReport, name="report"),
    path('index/Logout/', views.home),
    path('home/about/login',views.login),
    path('index/Logout/login', views.login),
    path('index/Logout/register', views.register),
    # path('index/report/dataset/', views.showFile),
    path('home/register/about/', views.about),
    path('home/register/about/login/', views.login),
    path('home/register/register/', views.register),
    path('home/register/login', views.login),
    path('logout/', views.logoutUser),
    path('home/login/home/login/register/',views.register,name="register"),
    path('index/report/list/',views.list),
    path('index/report/list/index/upload',views.uploadFile),
    path('index/report/list/Logout',views.home),
    path('index/report/list/login',views.login),
    path('home/about/login',views.login),
    path('home/about/home/login/register',views.register),
    # path('index/report/dataset/',views.output),
    # path('homeData',views.homeData,name='data'),
    path('index/prediction/',views.getPrediction),
    path('index/result',views.result,name='result')



]
