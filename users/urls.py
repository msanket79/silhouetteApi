from django.urls import path
from . import views,AdminViews,StudentViews,StaffViews,SecurityViews
from.models import student_profile
from django.shortcuts import render
from django.http import HttpResponse
from django.core.mail import send_mail
from silhouette.settings import EMAIL_HOST_USER
# //this is for testing purpose only
from django.utils import timezone
import datetime
from django.core.mail import send_mail
from silhouette.settings import EMAIL_HOST_USER
from rest_framework.views import APIView
from rest_framework import permissions
from rest_framework.response import Response


class demo(APIView):
    permission_classes=[permissions.AllowAny]
    def post(elf,request,format=None):
        
        # images = request.data.getlist('img')
        # confidence=request.data['confidence']
        # classifier.cosine_confidence=float(confidence)

       
        
        #     return Response()
        # ans=list(dict1.keys())
        # print(ans[0])
        ans=[]
        return Response({'rn1':ans[0]})
    



urlpatterns=[
    path('',views.LoginView.as_view(),name="login"),
    path('demo/',demo.as_view(),name="demo"),
    path('logout/',views.LogoutView.as_view(),name="logout"),
    path('change_password/',views.change_password.as_view(),name="change_password"),
    path('authenticated/',views.checkAuthenticated.as_view(),name="authenticated"),

    # admin urls
    path('admin_profile/',AdminViews.GetAdminProfileView.as_view(),name="admin_profile"),
    path('manage_student/',AdminViews.ManageStudents.as_view(),name="manage_students"),
    path('management/student/',AdminViews.UserAndStudentProfileCRUD.as_view(),name="student"),
    path('management/student/<str:pk>/',AdminViews.UserAndStudentProfileCRUD.as_view(),name="student1"),
    path('management/security/',AdminViews.UserAndSecurityProfileCRUD.as_view(),name="security"),
    path('management/security/<str:pk>/',AdminViews.UserAndSecurityProfileCRUD.as_view(),name="security1"),
    path('management/staff/',AdminViews.UserAndStaffProfileCRUD.as_view(),name="staff"),
    path('management/staff/<str:pk>/',AdminViews.UserAndStaffProfileCRUD.as_view(),name="staff1"),
    path('bulk_student_registration/',AdminViews.BulkStudentRegistration.as_view(),name="bulk_student_registration"),
    path('ban_student/',AdminViews.ban_student.as_view(),name="ban_student"),
    path('unban_student/',AdminViews.unban_student.as_view(),name="unban_student"),
    path('manage_security/',AdminViews.ManageSecurity.as_view(),name="manage_security"),
    path('manage_staff/',AdminViews.ManageStaff.as_view(),name="manage_staff"),
    path('delete_security/',AdminViews.DeleteSecurity.as_view(),name="delete_security"),
    path('add_students_for_outpass/',AdminViews.AddStudentsForOutpass.as_view(),name="add_students_for_swc"),



    #admin urls entry exit related data
    path('entry_exit_details/',AdminViews.EntryExitDetails.as_view(),name="entry_exit_details"),
    path('unban_requests/',AdminViews.UnbanRequests.as_view(),name="unban_requests"),
    path('delete_history/',AdminViews.DeleteHistory.as_view(),name="delete_history"),
    path('outpass_exits/',AdminViews.OutpassDetails.as_view(),name="outpass_exits"),


    # student page-------------------------------------------------->
    path('apply_outpass/',StudentViews.apply_outpass.as_view(),name="apply_outpass"),
    path('outpass_status/',StudentViews.outpass_status.as_view(),name="outpass_status"),
    path('appeal_ban/',StudentViews.appeal_ban.as_view(),name="appeal_ban"),
    path('my_entry_exit/',StudentViews.my_entry_exit.as_view(),name="my_entry_exit"),
    path('delete_ban/',StudentViews.delete_ban.as_view(),name="delete_ban"),
    path('student_profile/',StudentViews.StudentProfileView.as_view(),name="student_profile"),


    #staff page--------------------------------------------------------->
    path('outpass_requests/',StaffViews.outpass_requests.as_view(),name="outpass_requests"),
    path('staff_profile/',StaffViews.StaffProfileView.as_view(),name="staff_profile"),
    path('outpasses_approved/',StaffViews.OutpassesApproved.as_view(),name="outpasses_approved"),
    path('outpass_students_out/',StaffViews.OutpassStudentsOut.as_view(),name="outpass_students_out"),
    path('students_under_staff/',StaffViews.StudentsUnderStaff.as_view(),name="students_under_staff"),
    path('switch_staff_role/',StaffViews.SwitchStaffRole.as_view(),name="switch_staff_role"),






    # security page---------------------------------------------------------->
    path('security_profile/',SecurityViews.SecurityProfileView.as_view(),name="security_profile"),
    path('direct_entry/',SecurityViews.direct_entry.as_view(),name="direct_entry"),
    path('list_of_students_out/',SecurityViews.ListOfStudentsOutSerializer.as_view(),name='list_of_students_out'),
    path('direct_or_outpass/',SecurityViews.direct_or_outpass.as_view(),name="direct_or_outpass"),



]