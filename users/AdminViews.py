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
from outpass.models import staff_profile
import os
import pandas as pd
from django.conf import settings

# this is used to render the admin profile 
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
        



# This will be used to render all the entry exit details till now
class EntryExitDetails(APIView):
    def get(self,request,format=None):
        entry_exit_details=entry_exit.objects.all()
        entry_exit_details=EntryExitDetailsSerializer(entry_exit_details,many=True).data
        print(entry_exit_details)
        return Response({'entry_exit':entry_exit_details},content_type='application/json')



# this will be used show all the students who are out on the outpass
class OutpassDetails(APIView):
    def get(Self,request,format=None):
        entryexitdetails=entry_exit.objects.filter(entry_time__isnull=True).filter(outpass__isnull=False)
        entryexitdetails=EntryExitDetailsSerializer(entryexitdetails,many=True).data
        return Response(entryexitdetails)

# this will be used to render all the get all the unban appeals from user
# if we post to this view it can be used to approve the appeal 
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



# this is used to delete the history of the entry exits in a range
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
        

# this is used to create the profile of the student
class UserAndStudentProfileCreate(APIView):
    def post(self, request, format=None):
        serializer = CreateStudentSerializer(data=request.data)
        # print(request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({"success":"created successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# this is used to create the profile of the security
class UserAndSecurityProfileCreate(APIView):
    def post(self,request,format=None):
        serializer=CreateSecuritySerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({"user_id": user.id}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

# this is  used to create a staff profile -swc,fa,warden

class UserAndStaffProfileCreate(APIView):
    def post(self,request,format=None):
        serializer=CreateStaffSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
        
            return Response({"success":"successfully added staff",'id':str(user.id)}, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
# this is to manage the list of students and edit and delete the student data
class ManageStudents(APIView):
    def get(self,request,format=None):
        students=student_profile.objects.all()
        students=ManageStudentsSerializer(students,many=True).data
        return Response(students)
    

# this is to manage the list of security and edit and delete the security data
class ManageSecurity(APIView):
    def get(self,request,format=None):
        security_people=security_profile.objects.all()
        security_people=ManageSecuritySerializer(security_people,many=True).data
        return Response(security_people)

# this is used to delete the student data
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
# this is used to delete the security data
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
# this is used to unban a banned student from the manage student 
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
    # this is used to ban an unbanned student from the manage student
class ban_student(APIView):
    def post(self,request,format=None):
    
        id=self.request.data['id']
        student=student_profile.objects.get(id=id)
        
        student.ban=True
        student.save()
        return Response({'success':'student banned successfully'})
    

class AddStudentsForOutpass(APIView):
    def get(self,request,format=None):
        students=student_profile.objects.all()
        students=ManageStudentsSerializer(students,many=True).data
        return Response(students)
    def post(self,request,format=None):
        students=request.data['students']
        if students=="all":
            students1=student_profile.objects.all()
            print("all")
        elif students=="male":
            students1=student_profile.objects.filter(gender="male")
            print("male")
        elif students=="female":
            students1=student_profile.objects.filter(gender="female")
            print("female")
        else:
            students1=request.data.getlist('students')
            # print(students_list)
        if request.data.get('id'):
            staff_id=request.data['id']
        else:
            staff_id=request.user.staff_profile.id
        staff=staff_profile.objects.get(id=staff_id)
        if request.data.get('staff_type'):
            staff_type=request.data.get('staff_type')
        else:
            staff_type=staff.role
        if staff_type=="swc":
            staff.students_swc.add(*students1)
        elif staff_type=="fa":
            staff.students_fa.add(*students1)
        else:
            staff.students_warden.add(*students1)  
        return Response({'success':'students addded'})


class BulkStudentRegistration(APIView):
    def post(self,request,format=None):
        file1=request.FILES['file']
        print(file1)
        df=pd.read_excel(file1)
        completion_flag=0
        copied=0;
        total_rows=df.shape[0]
        for index, row in df.iterrows():
            flag=0
            try:
                print(row['NAME'])
                email=row['REG_NO']+"@iiitdwd.ac.in"
                user=customUser.objects.create_user(
                    username=row['REG_NO'],
                    email=email.lower(),
                    password="root",
                    user_type="3",
                )
                flag=1
                student1=student_profile.objects.create(
                    admin=user,
                    name=row['NAME'],
                    father=row['FATHERNAME'],
                    roll_no=row['REG_NO'],
                    phone_no=row['PHONE_NO'],
                    emergency_phone_no=row['EMERGENCY_NO'],
                    gender=row['GENDER'],
                    room_no=row['room no']
                )

                df.at[index, 'processed'] = 'Yes'
                copied+=1
            except Exception as e:
                completion_flag=1
                
                if flag==1:
                    user=customUser.objects.get(username=row['REG_NO'])
                    user.delete()
                pass
        # now saving the excell file
        file_name=file1.name
        new_file_path=os.path.join(settings.MEDIA_ROOT,f"processed_{file_name}")
        with pd.ExcelWriter(new_file_path) as writer:
            df.to_excel(writer, index=False)
        
        if completion_flag==1:
            return Response({'error':'only some_data is loaded','added':f"{copied}/{total_rows}"})         
        return Response({'success':'all students added','added':f"{copied}/{total_rows}"})

        






        
    