from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework import permissions
from rest_framework.response import Response
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie,csrf_exempt,csrf_protect
from .models import customUser
from django.contrib import auth
from users.EmailBackend import EmailBackend
from django.contrib.auth import login,logout
from django.http import JsonResponse
from rest_framework.authtoken.models import Token
from rest_framework import status

# Create your views here.


# this view is to get the csrftoken for the browser cookie



class checkAuthenticated(APIView):
    def get(self,request,format=None):
        user=self.request.user
        try:
            isAuthenticated=user.is_authenticated
            if isAuthenticated:
                return Response({'isAuthenticated':'success'})
            else:
                return Response({'isAuthenticated':'error'})
        except:
            return Response({'error':"something went wrong while checking for the authentication of the user"})
        
    
class LoginView(APIView):
    permission_classes=(permissions.AllowAny,)
    def post(self,request,format=None):
        data=self.request.data
        email=data['email']
        password=data['password']
        user=EmailBackend.authenticate(request,username=email,password=password)
        if user is not None:
            token, created = Token.objects.get_or_create(user=user)
            # auth.login(request,user)
            data={'success':"User authenticated",'id':user.id}
            data['token'] = token.key
            # //if he is admin
            if user.user_type=="1":
                data['admin']="admin"

            # if user is security
            if user.user_type=="2":
                if user.security_profile.head:
                    data['security']="head"
                else:
                    data['security']="normal"

            #if user is student
            if user.user_type=="3":
                data['student']="student"

            #if user is staff
            if user.user_type=="4":
                if user.staff_profile.Permission_level=="fa":
                    data['staff']="fa"
                if user.staff_profile.Permission_level=="warden":
                    data['staff']="warden"
                else:
                    data['staff']="swc"


            return Response(data)
        else:
            return Response({'error':'Error Authentication'})
        
class LogoutView(APIView):

    def post(self, request):
        try:
            request.user.auth_token.delete()
            return Response({"message": "Logout successful"}, status=status.HTTP_200_OK)
        except:
            return Response({'error':'logout error'})

# class LogoutView(APIView):
#     def post(self,request,format=None):
#         try:
#             auth.logout(request)
#             return Response({'success':'logout success'})
#         except:
#             return Response({'error':'Something went wrong while logging out the user'})
        
        
class change_password(APIView):
    def post(self,request,format=None):
        data=self.request.data
        curr_pass=data['old_password']
        new_pass=data['new_password']
        email=self.request.user.email
        print(email)
        user=EmailBackend.authenticate(self.request,username=email,password=curr_pass)
        if user:
            user=customUser.objects.get(email=email)
            user.set_password(new_pass)
            user.save()
            return Response({"success":"change password for the user successfull"})
        else:
            return Response({"error":"Please enter the correct password"})
        


