from rest_framework import serializers
from .models import admin_profile,customUser,entry_exit,student_profile,security_profile

from outpass.models import staff_profile,Outpass

# serialzers for admin page--------------------------------------------------------------
class AdminProfileSerializer(serializers.ModelSerializer):
    email=serializers.CharField(source='admin.email')
    profile_pic=serializers.ImageField()
    class Meta:
       
        model=admin_profile
        fields = ['name','phone_no','profile_pic','email']
class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model=customUser
        fields="__all__"

class EntryExitDetailsSerializer(serializers.ModelSerializer):
    roll_no=serializers.CharField(source='roll_no.roll_no')
    name=serializers.CharField(source='roll_no.name')
    phone_no=serializers.CharField(source='roll_no.phone_no')

    
    class Meta:
        model=entry_exit
        fields=['id','exit_time','entry_time','roll_no','name','phone_no']


class ManageStudentsSerializer(serializers.ModelSerializer):
    class Meta:
        model=student_profile
        fields=['name','roll_no','father','phone_no','emergency_phone_no','gender','ban','id']
class ManageSecuritySerializer(serializers.ModelSerializer):
    class Meta:
        model=security_profile
        fields=['name','phone_no','id']

class UnbanRequestsSerializer(serializers.ModelSerializer):
    reason=serializers.CharField(source='appeal_unban.reason')
    class Meta:
        model=student_profile
        fields=['roll_no','name','reason','phone_no']

class CreateStudentSerializer(serializers.ModelSerializer):
    email=serializers.EmailField(source='admin.email')
    password=serializers.CharField(source='admin.password')
    profile_pic=serializers.ImageField(max_length=None,use_url=True)
    class Meta:
        model=student_profile
        fields=['email','password','name','roll_no','father','phone_no','emergency_phone_no','gender','profile_pic']
    def create(self,validated_data):
        user_data=validated_data.pop('admin')
        try:
            user=customUser.objects.create_user(
                username=validated_data['roll_no'],
                email=user_data['email'],
                password=user_data['password'],
                user_type='3'
            )
        except:
            raise serializers.ValidationError('User already exists', code='error')

        profile_pic=validated_data.pop('profile_pic')
        student_profile1=student_profile.objects.create(
            admin=user,**validated_data
        )
        student_profile1.profile_pic=profile_pic
        student_profile1.save()
        return student_profile1
    def to_internal_value(self, data):
        try:
            return super().to_internal_value(data)
        except serializers.ValidationError as exc:
            errors = {}
            for field, error_list in exc.detail.items():
                errors['error'] = errors.get('error', []) + error_list
            raise serializers.ValidationError(errors)
        

class CreateSecuritySerializer(serializers.ModelSerializer):
    email=serializers.CharField(source='admin.email')
    password=serializers.CharField(source='admin.password')
    profile_pic=serializers.ImageField(max_length=None,use_url=True)
    class Meta:
        model=security_profile
        fields=['email','password','name','phone_no','profile_pic']
    def create(self,validated_data):
        user_data=validated_data.pop('admin')
        try:
            user=customUser.objects.create_user(
                username=user_data['email'],
                email=user_data['email'],
                password=user_data['password'],
                user_type='2'
            )
        except:
            raise serializers.ValidationError('User already exists', code='error')
        profile_pic=validated_data.pop('profile_pic')
        security_profile1=security_profile.objects.create(admin=user,**validated_data)
        security_profile1.profile_pic=profile_pic
        security_profile1.save()
        return security_profile1
    def to_internal_value(self, data):
        try:
            return super().to_internal_value(data)
        except serializers.ValidationError as exc:
            errors = {}
            for field, error_list in exc.detail.items():
                errors['error'] = errors.get('error', []) + error_list
            raise serializers.ValidationError(errors)

class CreateStaffSerializer(serializers.ModelSerializer):
    email=serializers.CharField(source='admin.email')
    password=serializers.CharField(source='admin.password')
    profile_pic=serializers.ImageField(max_length=None,use_url=True)
    class Meta:
        model=staff_profile
        fields=['email','password','profile_pic','name','phone_no']
    def create(self,validated_data):
        print(validated_data)
        user_data=validated_data.pop('admin')
        try:
            user=customUser.objects.create_user(
                username=user_data['email'],
                email=user_data['email'],
                password=user_data['password'],
                user_type='4'
            )
        except:
            raise serializers.ValidationError('User already exists', code='error')
        profile_pic=validated_data.pop('profile_pic')
        staff_profile1=staff_profile.objects.create(admin=user,**validated_data)
        staff_profile1.profile_pic=profile_pic
        staff_profile1.save()
        return staff_profile1
    def to_internal_value(self, data):
        try:
            return super().to_internal_value(data)
        except serializers.ValidationError as exc:
            errors = {}
            for field, error_list in exc.detail.items():
                errors['error'] = errors.get('error', []) + error_list
            raise serializers.ValidationError(errors)


# student page serializers -------------------------------------------------------------------------------------

class StudentProfileSerializer(serializers.ModelSerializer):
    email=serializers.CharField(source='admin.email')
    profile_pic=serializers.ImageField()
    class Meta:
        model=student_profile
        fields = ['name','roll_no','father','phone_no','emergency_phone_no','gender','profile_pic','email']



class OutpassSerialzer(serializers.ModelSerializer):
    roll_no=serializers.CharField(source='roll_no.roll_no')
    class Meta:
        model=Outpass
        fields=['id','From','To','Reason','faculty_approval','warden_approval','swc_approval','approved','roll_no']

class MyEntryExitSerialzer(serializers.ModelSerializer):
    class Meta:
        model=entry_exit
        fields=['id','entry_time','exit_time']

    


# staff page serializers ---------------------------------------------------------------------------------------
class OutpassRequestsSerialzer(serializers.ModelSerializer):
    roll_no=serializers.CharField(source='roll_no.roll_no')
    name=serializers.CharField(source='roll_no.name')
    phone_no=serializers.CharField(source='roll_no.phone_no')
    ban=serializers.BooleanField(source='roll_no.ban')
    class Meta:
        model=Outpass
        fields=['id','From','To','Reason','roll_no','name','phone_no','ban']

class StaffProfileSerializer(serializers.ModelSerializer):
    email=serializers.CharField(source='admin.email')
    profile_pic=serializers.ImageField()
    class Meta:
        model=staff_profile
        fields=['name','phone_no','profile_pic','Permission_level','email']



#security page serializers
class SecurityProfileSerializer(serializers.ModelSerializer):
    email=serializers.CharField(source='admin.email')
    profile_pic=serializers.ImageField()
    class Meta:
        model=security_profile
        fields=['name','phone_no','profile_pic','email','head']


class ScannedStudentSerializer(serializers.ModelSerializer):
    profile_pic=serializers.ImageField()
    class Meta:
        model=student_profile
        fields = ['name','roll_no','profile_pic']





