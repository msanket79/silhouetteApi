from django.db import models
from users.models import student_profile,customUser,security_profile,admin_profile,entry_exit
from django.dispatch import receiver
from django.db.models.signals import post_save

# # Create your models here.
class Outpass(models.Model):
    id=models.AutoField(primary_key=True)
    roll_no=models.ForeignKey(student_profile,on_delete=models.CASCADE)
    From=models.DateField()
    To=models.DateField()
    Reason=models.TextField(max_length=1000,null=True,blank=True)
    faculty_approval=models.BooleanField(blank=True,null=True)
    warden_approval=models.BooleanField(blank=True,null=True)
    swc_approval=models.BooleanField(blank=True,null=True,default=True)
    approved=models.BooleanField(null=True,blank=True)
    entry=models.OneToOneField(entry_exit,on_delete=models.CASCADE,null=True,blank=True)
    used=models.BooleanField(null=True,blank=True)

class staff_profile(models.Model):
    id=models.AutoField(primary_key=True)
    name=models.CharField(max_length=200)
    profile_pic=models.FileField(max_length=400,default='media/default.jpg')
    phone_no=models.CharField(max_length=100)
    Permission_level=models.CharField(max_length=50)
    students=models.ManyToManyField(student_profile)
    admin=models.OneToOneField(customUser,on_delete=models.CASCADE)



@receiver(post_save,sender=customUser)
def create_user_profile(sender,instance,created,**kwargs):
    if created:
        if instance.user_type==1:
            admin_profile.objects.create(admin=instance)
            print("profile created using the trigger")
        if instance.user_type==2:
            security_profile.objects.create(admin=instance)
            print("profile created using the trigger")
        if instance.user_type==3:
            student_profile.objects.create(admin=instance)
            print("profile created using the trigger")
        if instance.user_type==4:
            staff_profile.objects.create(admin=instance)
            print("profile created using the trigger")

@receiver(post_save,sender=customUser)
def save_user_profile(instance,created,**kwargs):
    if instance.user_type==1:
        instance.admin_profile.save()
        print("profile saved using the trigger")
    if instance.user_type==2:
        instance.security_profile.save()
        print("profile saved using the trigger")
    if instance.user_type==3:
        instance.student_profile.save()
        print("profile saved using the trigger")
    if instance.user_type==4:
        instance.staff_profile.save()
        print("profile saved using the trigger")