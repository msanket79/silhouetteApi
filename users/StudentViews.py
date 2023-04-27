from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework import permissions
from rest_framework.response import Response
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie,csrf_exempt,csrf_protect
from django.contrib.auth.models import User
from django.contrib import auth
from .serializers import AdminProfileSerializer,UserProfileSerializer,EntryExitDetailsSerializer,UnbanRequestsSerializer,CreateStudentSerializer,CreateSecuritySerializer,CreateStaffSerializer,OutpassSerialzer,MyEntryExitSerialzer,StudentProfileSerializer
from .models import admin_profile,customUser,entry_exit,student_profile,appeal_unban
from .EmailBackend import EmailBackend
from rest_framework import status
from outpass.models import Outpass
from django.db.models import Q
import datetime

class apply_outpass(APIView):
    def post(self,request,format=None):
        data=request.data
        roll_no=request.user.username
        student=student_profile.objects.get(roll_no=roll_no)
        query=(Q(roll_no=student) & ~Q(used=True))
        data1=Outpass.objects.filter(query)
        if data1:
            return Response({'error':'already applied for the outpass'})
        from_date =datetime.datetime.strptime(data['from_date'], '%Y-%m-%d')
        to_date = datetime.datetime.strptime(data['to_date'], '%Y-%m-%d')
        if from_date<to_date:
            
            reason=data['reason']
        
            
            outpass=Outpass.objects.create(roll_no=student,From=from_date,To=to_date,Reason=reason)

            #if chuttiya is greater than 10 set the swc flag to None
            gap=to_date-from_date
            if datetime.timedelta(days=10)<=gap:
                outpass.swc_approval=None
                print(outpass.swc_approval)
                
            outpass.save()
            return Response({'success':'successfully applied for the outpass'})
        else:
            return Response({'error':'enter valid date'})


class outpass_status(APIView):
    def get(self,request,format=None):
        student=student_profile.objects.get(roll_no=request.user.username)
        outpass=Outpass.objects.filter((Q(roll_no=student) & ~Q(used=True)))
        outpass=OutpassSerialzer(outpass,many=True).data
        return Response(outpass)
    def post(self,request,format=None):
        pk=request.data['id']
        try:
            outpass=Outpass.objects.get(id=pk)
            outpass.delete()
            return Response({'success':'successfully deleted outpass'})
        except:
            return Response({'error':'some problem occured'})

class appeal_ban(APIView):
    def get(self,request,format=None):
        user=request.user
        student=student_profile.objects.get(admin=user)
        
        if student.ban:
           
            try:
                if student.appeal_unban: 
                    appeal=appeal_unban.objects.get(student_id=student)
                    return Response({'appeal':'true','roll_no':str(user.username),'reason':str(appeal.reason)})
            except:
                return Response({'ban':'true'})
            
        else:
            return Response({'error':'student is not banned'})


    def post(self,request,format=None):
        reason=request.POST['reason']
        roll_no=request.user.username
        try:
            student=student_profile.objects.get(roll_no=roll_no)
            req=appeal_unban.objects.create(reason=reason,student_id=student)
            req.save()
            return Response({'success':"successfully applied of unban"})
        except Exception as e:
            print(e)
            return Response({'error':"already applied for unban"})
class delete_ban(APIView):
    def post(self,request,format=None):
        user=request.user
        student=student_profile.objects.get(admin=user)
        student.appeal_unban.delete()
        student.save()
        return Response({'success':'successfully deleted the unban request'})
    

class my_entry_exit(APIView):
    def get(self,request,format=None):
        student=student_profile.objects.get(roll_no=request.user.username)
        entry_exit1=entry_exit.objects.filter(roll_no=student)
        entry_exit1=MyEntryExitSerialzer(entry_exit1,many=True).data
        return Response(entry_exit1)
class StudentProfileView(APIView):
    def get(self,request,format=None):
        user=request.user
        student=student_profile.objects.get(admin=user)
        student=StudentProfileSerializer(student).data
        return Response(student)
    


        
    
