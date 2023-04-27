from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework import permissions
from rest_framework.response import Response
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie,csrf_exempt,csrf_protect
from django.contrib.auth.models import User
from django.contrib import auth
from .serializers import AdminProfileSerializer,UserProfileSerializer,EntryExitDetailsSerializer,UnbanRequestsSerializer,CreateStudentSerializer,CreateSecuritySerializer,CreateStaffSerializer,ManageStudentsSerializer,ManageSecuritySerializer
from .models import admin_profile,customUser,entry_exit,student_profile,security_profile
from .EmailBackend import EmailBackend
from rest_framework import status

class GetAdminProfileView(APIView):
    def get(self,request,format=None):
        try:
            user=self.request.user
            print(user)
            admin_profile1=admin_profile.objects.get(admin=user)
            print(admin_profile1)
            admin_profile1=AdminProfileSerializer(admin_profile1).data
            return Response(admin_profile1)
        except Exception as e:
            print(e)
        return Response({'error':'something went wrong while retreiving the admin profile'})    
        

class EntryExitDetails(APIView):
    def get(self,request,format=None):
        entry_exit_details=entry_exit.objects.all()
        entry_exit_details=EntryExitDetailsSerializer(entry_exit_details,many=True).data
        print(entry_exit_details)
        return Response({'entry_exit':entry_exit_details},content_type='application/json')

class OutpassDetails(APIView):
    def get(Self,request,format=None):
        entryexitdetails=entry_exit.objects.filter(entry_time__isnull=True).filter(outpass__isnull=False)
        entryexitdetails=EntryExitDetailsSerializer(entryexitdetails,many=True).data
        return Response(entryexitdetails)

# @method_decorator(csrf_protect,name="dispatch")  
class UnbanRequests(APIView):
    def get(self,request,format=None):
        students=student_profile.objects.filter(ban=True).filter(appeal_unban__isnull=False)
        students=UnbanRequestsSerializer(students,many=True).data
        print(students)
        return Response({'students':students})
    def post(self,request,format=None):
    
        roll_no=self.request.data['roll_no']
        student=student_profile.objects.get(roll_no=roll_no)
        try:
            student.appeal_unban.delete()
        except:
            pass
        student.ban=False
        student.save()
        return Response({'success':'student unbanned successfully'})

class DeleteHistory(APIView):
    permission_classes=[permissions.AllowAny]
    def post(self,request,format=None):
        
        data=self.request.data
        from_date=data['from_date']
        to_date=data['to_date']
        password=data['password']
        email=self.request.user
        # print(email)
        
        if EmailBackend.authenticate(self.request,username=email,password=password):
            # ans=entry_exit.objects.filter(entry_time__gte=from_date,
            #                 entry_time__lte=to_date).delete()
            return Response({'success':'history deleted successfully'})
        else:
            return Response({'error':'password is not correct'})
        


class UserAndStudentProfileCreate(APIView):
    def post(self, request, format=None):
        serializer = CreateStudentSerializer(data=request.data)
        # print(request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({"success":"created successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class UserAndSecurityProfileCreate(APIView):
    def post(self,request,format=None):
        serializer=CreateSecuritySerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({"user_id": user.id}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
class UserAndStaffProfileCreate(APIView):
    def post(self,request,format=None):
        serializer=CreateStaffSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({"success":"successfully added staff"}, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

class ManageStudents(APIView):
    def get(self,request,format=None):
        students=student_profile.objects.all()
        students=ManageStudentsSerializer(students,many=True).data
        return Response(students)

class ManageSecurity(APIView):
    def get(self,request,format=None):
        security_people=security_profile.objects.all()
        security_people=ManageSecuritySerializer(security_people,many=True).data
        return Response(security_people)

class DeleteStudent(APIView):
    def post(self,request,foramt=None):
        id=request.data['id']
        try:
            student=student_profile.objects.get(id=id)
            student.admin.delete()
            student.delete()
            return Response({'success':"successfully deleted the student"})
        except:
            return Response({'error':'some error occured'})
class DeleteSecurity(APIView):
    def post(Self,reqquest,format=None):
        id=reqquest.data['id']
        try:
            security=security_profile.objects.get(id=id)
            security.admin.delete()
            security.delete()
            return Response({'success':"successfully deleted the security"})
        except:
            return Response({'error':'some error occured'})
class unban_student(APIView):
    def post(self,request,format=None):
    
        id=self.request.data['id']
        student=student_profile.objects.get(id=id)
        try:
            student.appeal_unban.delete()
        except:
            pass
        student.ban=False
        student.save()
        return Response({'success':'student unbanned successfully'})
class ban_student(APIView):
    def post(self,request,format=None):
    
        id=self.request.data['id']
        student=student_profile.objects.get(id=id)
        
        student.ban=True
        student.save()
        return Response({'success':'student banned successfully'})
        






        
    