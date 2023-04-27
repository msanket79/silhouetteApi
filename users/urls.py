from django.urls import path
from . import views,AdminViews,StudentViews,StaffViews,SecurityViews


urlpatterns=[
    path('',views.LoginView.as_view(),name="login"),
    
    path('logout/',views.LogoutView.as_view(),name="logout"),
    path('change_password/',views.change_password.as_view(),name="change_password"),
    path('authenticated/',views.checkAuthenticated.as_view(),name="authenticated"),






    # admin urls
    path('admin_profile/',AdminViews.GetAdminProfileView.as_view(),name="admin_profile"),
    path('manage_student/',AdminViews.ManageStudents.as_view(),name="manage_students"),
    path('ban_student/',AdminViews.ban_student.as_view(),name="ban_student"),
    path('unban_student/',AdminViews.unban_student.as_view(),name="unban_student"),
    path('manage_security/',AdminViews.ManageSecurity.as_view(),name="manage_security"),

    #amdin page-student related
    path('create_student/',AdminViews.UserAndStudentProfileCreate.as_view(),name="create_student"),
    path('delete_student/',AdminViews.DeleteStudent.as_view(),name="delete_student"),
    path('delete_security/',AdminViews.DeleteSecurity.as_view(),name="delete_security"),



    #admin page-security related
    path('create_security/',AdminViews.UserAndSecurityProfileCreate.as_view(),name="create_security"),




    #admin page-staff related
    path('create_staff/',AdminViews.UserAndStaffProfileCreate.as_view(),name="create_staff"),


    #admin urls entry exit related data
    path('entry_exit_details/',AdminViews.EntryExitDetails.as_view(),name="entry_exit_details"),
    path('unban_requests/',AdminViews.UnbanRequests.as_view(),name="unban_requests"),
    path('delete_history/',AdminViews.DeleteHistory.as_view(),name="delete_history"),
    path('outpass_exits/',AdminViews.OutpassDetails.as_view(),name="outpass_exits"),
    # path('unban_student/',AdminViews.UnbanStudent.as_view(),name="unban_student"),





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




    # security page---------------------------------------------------------->
    path('security_profile/',SecurityViews.SecurityProfileView.as_view(),name="security_profile"),
    path('direct_entry/',SecurityViews.direct_entry.as_view(),name="direct_entry"),
    path('list_of_students_out/',SecurityViews.ListOfStudentsOutSerializer.as_view(),name='list_of_students_out'),
    path('direct_or_outpass/',SecurityViews.direct_or_outpass.as_view(),name="direct_or_outpass"),




    
    




]