from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework import permissions
from rest_framework.response import Response
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie,csrf_exempt,csrf_protect
from django.contrib.auth.models import User
from django.contrib import auth
from outpass.models import staff_profile,Outpass
from .serializers import AdminProfileSerializer,OutpassRequestsSerialzer,StaffProfileSerializer

from rest_framework import status

class outpass_requests(APIView):
    def get(self,request,format=None):
        staff_id=request.user.staff_profile.id
        staff=staff_profile.objects.get(id=staff_id)
        staff_type=staff.Permission_level
        students=staff.students.all()
        s1=[]     
        for student in students:
        
            if staff_type=="swc":
                outpasses=Outpass.objects.filter(roll_no=student).filter(swc_approval__isnull=True)
            if staff_type=="warden":
                outpasses=Outpass.objects.filter(roll_no=student).filter(warden_approval__isnull=True)
            
            if staff_type=="fa":
                outpasses=Outpass.objects.filter(roll_no=student).filter(faculty_approval__isnull=True)

            outpass=OutpassRequestsSerialzer(outpasses,many=True).data
            print(type(outpass))
            # need to revist
            s1.extend(outpass)
        return  Response(s1)
    def post(self,request,format=None):
        staff_type=request.user.staff_profile.Permission_level
        print(staff_type)
        pk=request.data['id']
        if request.data['status']=='Approve':
            try:
                outpass=Outpass.objects.get(id=pk)
                print(outpass.id)
                if staff_type=="swc":
                    outpass.swc_approval=True
                if staff_type=="warden":
                    outpass.warden_approval=True
                if staff_type=="fa":
                    print("again in fa")
                    outpass.faculty_approval=True
                outpass.save()
                if outpass.swc_approval and outpass.warden_approval and outpass.swc_approval:
                    outpass.approved=True
                    outpass.save()
                    #here we will send a mail to the student
                return Response({'success':'outpasss accepted successfully'})
                
            except:
                return Response({'error':'outpass not valid'})
        else:
            try:
                outpass=Outpass.objects.get(id=pk)
                outpass.delete()
                return Response({'success':'outpass deleted'})
            except:
                return Response({'error':'outpass does not exist'})


class StaffProfileView(APIView):
    def get(self,request,format=None):
        user=request.user
        staff=staff_profile.objects.get(admin=user)
        staff=StaffProfileSerializer(staff).data
        return Response(staff)
            