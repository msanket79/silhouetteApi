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
    email=serializers.EmailField(source="admin.email")
    class Meta:
        model=student_profile
        fields=['name','roll_no','father','phone_no','emergency_phone_no','gender','ban','id','email']
class ManageStaffsSerializer(serializers.ModelSerializer):
    email=serializers.EmailField(source="admin.email")
    class Meta:
        model=staff_profile
        fields=['name','phone_no','fa','warden','swc','id','email']
class ManageSecuritySerializer(serializers.ModelSerializer):
    class Meta:
        model=security_profile
        fields=['name','phone_no','id']

class UnbanRequestsSerializer(serializers.ModelSerializer):
    reason=serializers.CharField(source='appeal_unban.reason')
    class Meta:
        model=student_profile
        fields=['roll_no','name','reason','phone_no']

class StudentSerializer(serializers.ModelSerializer):
    email=serializers.EmailField(source='admin.email')
    password=serializers.CharField(source='admin.password',write_only=True,required=False)
    profile_pic=serializers.ImageField(max_length=None,use_url=True,required=False)
    class Meta:
        model=student_profile
        fields=['email','password','name','roll_no','father','phone_no','emergency_phone_no','gender','profile_pic','room_no']
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
        student_profile1=student_profile.objects.create(
            admin=user,**validated_data
        )
        if validated_data.get('profile_pic'):
            profile_pic=validated_data.pop('profile_pic')
            student_profile1.profile_pic=profile_pic
        student_profile1.save()
        return student_profile1

    def update(self, instance, validated_data):
        user_data = validated_data.pop('admin')
        user = instance.admin
        print(validated_data)
        try:
            if user_data.get('email'):
                user.email = user_data['email']

            if user_data.get('password'):
                user.set_password(user_data['password'])

            user.save()
            if validated_data.get('profile_pic'):
                instance.profile_pic = validated_data['profile_pic']

            instance.name = validated_data.get('name', instance.name)
            instance.roll_no = validated_data.get('roll_no', instance.roll_no)
            instance.room_no = validated_data.get('roll_no', instance.room_no)
            instance.father = validated_data.get('father', instance.father)
            instance.phone_no = validated_data.get('phone_no', instance.phone_no)
            instance.emergency_phone_no = validated_data.get('emergency_phone_no', instance.emergency_phone_no)
            instance.gender = validated_data.get('gender', instance.gender)

            instance.save()
        except:
            raise serializers.ValidationError('some error occured', code='error')
        return instance
    
    def to_internal_value(self, data):
        try:
            return super().to_internal_value(data)
        except serializers.ValidationError as exc:
            errors = {}
            for field, error_list in exc.detail.items():
                errors['error'] = errors.get('error', []) + error_list
            raise serializers.ValidationError(errors)

        



class SecuritySerializer(serializers.ModelSerializer):
    email=serializers.CharField(source='admin.email')
    password=serializers.CharField(source='admin.password',write_only=True)
    profile_pic=serializers.ImageField(max_length=None,use_url=True,required=False)
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
        security_profile1=security_profile.objects.create(admin=user,**validated_data)
        if validated_data.get('profile_pic'):
            profile_pic=validated_data.pop('profile_pic')
            security_profile1.profile_pic=profile_pic
        security_profile1.save()
        
        return security_profile1
    def update(self, instance, validated_data):
        # Update the user email and password
        try:
            user_data = validated_data.pop('admin')
            user = instance.admin
            user.email = user_data.get('email', user.email)
            user.set_password(user_data.get('password', user.password))
            user.save()

            # Update the security profile fields
            instance.name = validated_data.get('name', instance.name)
            instance.phone_no = validated_data.get('phone_no', instance.phone_no)
            instance.profile_pic = validated_data.get('profile_pic', instance.profile_pic)
            instance.save()
        except:
            raise serializers.ValidationError('some error occured', code='error')

        return instance

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
    profile_pic=serializers.ImageField(max_length=None,use_url=True,required=False)
    swc = serializers.CharField(required=False)
    warden = serializers.CharField(required=False)
    fa = serializers.CharField(required=False)
    class Meta:
        model=staff_profile
        fields=['email','password','profile_pic','name','phone_no','gender','swc','warden','fa']
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
        swc=False
        warden=False
        fa=False
        role="fa"
        if validated_data.get('swc'):
            print('swc get kia hai aur swc ko true kia')
            role="swc"
            swc=True
            validated_data.pop('swc')
        if validated_data.get('warden'):
            print('warden get kia hai aur warden ko true kia')
            role="warden"
            warden=True
            validated_data.pop('warden')
        if validated_data.get('fa'):
            print('fa get kia hai aur fa ko true kia')
            role="fa"
            fa=True
            validated_data.pop('fa')

        staff_profile1=staff_profile.objects.create(admin=user,**validated_data)
        print(swc)
        if swc:
            staff_profile1.swc=True
        print(warden)
        if warden:
            staff_profile1.warden=True
        print(fa)
        if fa:
            staff_profile1.fa=True
        staff_profile1.role=role
        print(role)
        if validated_data.get('profile_pic'):
            profile_pic=validated_data.pop('profile_pic')
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

class UpdateStaffSerializer(serializers.ModelSerializer):
    email = serializers.CharField(source='admin.email')
    password = serializers.CharField(source='admin.password')
    profile_pic = serializers.ImageField(max_length=None, use_url=True)
    swc = serializers.CharField(required=False)
    warden = serializers.CharField(required=False)
    fa = serializers.CharField(required=False)

    class Meta:
        model = staff_profile
        fields = ['email', 'password', 'profile_pic', 'name', 'phone_no','role','gender','warden','fa','swc']

    def update(self, instance, validated_data):
        # Update the user email and password
        try:
            user_data = validated_data.pop('admin')
            user = instance.admin
            user.email = user_data.get('email', user.email)
            user.set_password(user_data.get('password', user.password))
            user.save()
            swc=False
            warden=False
            fa=False
            role="fa"
            if validated_data.get('swc'):
                role="swc"
                swc=True
                validated_data.pop('swc')
            if validated_data.get('warden'):
                role="warden"
                warden=True
                validated_data.pop('warden')
            if validated_data.get('fa'):
                role="fa"
                fa=True
                validated_data.pop('fa')
            # Update the staff profile fields
            instance.fa=validated_data.get('fa',instance.fa)
            instance.warden=validated_data.get('fa',instance.warden)
            instance.swc=validated_data.get('fa',instance.swc)
            instance.name = validated_data.get('name', instance.name)
            instance.gender=validated_data.get('gender',instance.gender)
            instance.phone_no = validated_data.get('phone_no', instance.phone_no)
            instance.profile_pic = validated_data.get('profile_pic', instance.profile_pic)
            instance.save()
        except:
            raise serializers.ValidationError('some error occured', code='error')

        return instance

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
        fields=['name','phone_no','profile_pic','role','email']



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





