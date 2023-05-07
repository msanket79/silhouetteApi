from django.db import models
from django.contrib.auth.models import AbstractUser
from django.dispatch import receiver
from django.db.models.signals import post_save

# from outpass.models import staff_profile

# Create your models here.


class customUser(AbstractUser):

    email = models.EmailField(unique=True)
    user_type_data=((1,'admin'),(2,'security'),(3,'student'),(4,'staff'),)
    user_type=models.CharField(default=1,choices=user_type_data,max_length=100)


class student_profile(models.Model):
    id=models.AutoField(primary_key=True)
    name=models.CharField(max_length=200)
    roll_no=models.CharField(unique=True,max_length=100)
    father=models.CharField(max_length=200)
    phone_no=models.CharField(max_length=100)
    emergency_phone_no=models.CharField(max_length=100)
    gender=models.CharField(max_length=100,default="male")
    profile_pic=models.FileField(max_length=400,default="media/default.jpg")
    room_no=models.CharField(default="342",max_length=10)
    ban=models.BooleanField(null=True,blank=True,default=False)
    objects=models.Manager()
    admin=models.OneToOneField(customUser,on_delete=models.CASCADE)
class appeal_unban(models.Model):
    reason=models.TextField(max_length=1000,null=True,blank=True)
    student_id=models.OneToOneField(student_profile,null=True,blank=True,on_delete=models.CASCADE)
    cause=models.TextField(max_length=1000)

class security_profile(models.Model):
    id=models.AutoField(primary_key=True)
    phone_no=models.CharField(max_length=100)
    name=models.CharField(max_length=200)
    profile_pic=models.FileField(max_length=400,default="media/default.jpg")
    head=models.BooleanField(default=False,blank=True)
    admin=models.OneToOneField(customUser,on_delete=models.CASCADE)
    objects=models.Manager()

class admin_profile(models.Model):
    id=models.AutoField(primary_key=True)
    name=models.CharField(max_length=200)
    phone_no=models.CharField(max_length=200,null=True,blank=True)
    profile_pic=models.FileField(max_length=400,default="media/default.jpg")
    admin=models.OneToOneField(customUser,on_delete=models.CASCADE)
    objects=models.Manager()

class entry_exit(models.Model):
    id=models.AutoField(primary_key=True)
    roll_no=models.ForeignKey(student_profile,on_delete=models.CASCADE,null=True,blank=True)
    exit_time=models.DateTimeField(auto_now_add=True)
    entry_time=models.DateTimeField(null=True,blank=True)
    objects=models.Manager()




