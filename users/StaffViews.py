from rest_framework.views import APIView
from rest_framework.response import Response
from outpass.models import staff_profile,Outpass
from .serializers import OutpassRequestsSerialzer,StaffProfileSerializer,EntryExitDetailsSerializer,ManageStudentsSerializer
from .models import entry_exit,student_profile
from .permissions import IsStaff


class StaffViewPermission(APIView):
    permission_classes=[IsStaff]

# this will fetch all the all the outpasses when we do a get request and on post we can delete and approve the outpass 
class outpass_requests(StaffViewPermission):
    def get(self,request,format=None):
        staff_id=request.user.staff_profile.id
        staff=staff_profile.objects.get(id=staff_id)
        staff_type=staff.role
        if staff_type=="swc":
            students=staff.students_swc.all()
        elif staff_type=="warden":
            students=staff.students_warden.all()
        else:
            students=staff.students_fa.all()
        s1=[]     
        for student in students:
        
            if staff_type=="swc":
                outpasses=Outpass.objects.filter(roll_no=student).filter(swc_approval__isnull=True)
            if staff_type=="warden":
                outpasses=Outpass.objects.filter(roll_no=student).filter(warden_approval__isnull=True)
            
            if staff_type=="fa":
                outpasses=Outpass.objects.filter(roll_no=student).filter(faculty_approval__isnull=True)

            outpass=OutpassRequestsSerialzer(outpasses,many=True).data
            # need to revist
            s1.extend(outpass)
        return  Response(s1)
    def post(self,request,format=None):
        staff_type=request.user.staff_profile.role
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
                # send mail here
                outpass.delete()
                return Response({'success':'outpass deleted'})
            except:
                return Response({'error':'outpass does not exist'})

# this will be used to render the profile view
class StaffProfileView(StaffViewPermission):
    def get(self,request,format=None):
        user=request.user
        staff=staff_profile.objects.get(admin=user)
        staff=StaffProfileSerializer(staff).data
        return Response(staff)
    
class OutpassesApproved(StaffViewPermission):
    def get(self,request,format=None):
        staff_id=request.user.staff_profile.id
        staff=staff_profile.objects.get(id=staff_id)
        staff_type=staff.role
        if staff_type=="swc":
            students=staff.students_swc.all()
        elif staff_type=="warden":
            students=staff.students_warden.all()
        else:
            students=staff.students_fa.all()
        s1=[]     
        for student in students:
        
            if staff_type=="swc":
                outpasses=Outpass.objects.filter(roll_no=student).filter(swc_approval__isnull=False).exclude(used=True)
            if staff_type=="warden":
                outpasses=Outpass.objects.filter(roll_no=student).filter(warden_approval__isnull=False).exclude(used=True)
            
            if staff_type=="fa":
                outpasses=Outpass.objects.filter(roll_no=student).filter(faculty_approval__isnull=False).exclude(used=True)

            outpass=OutpassRequestsSerialzer(outpasses,many=True).data
            # need to revist
            s1.extend(outpass)
        return  Response(s1)

    def post(self,request,format=None):
        staff_type=request.user.staff_profile.role
        pk=request.data['id']
        try:
            outpass=Outpass.objects.get(id=pk)
            print(outpass.id)
            if staff_type=="swc":
                outpass.swc_approval=None
            if staff_type=="warden":
                outpass.warden_approval=None
            if staff_type=="fa":
                print("again in fa")
                outpass.faculty_approval=None
            outpass.save()
                #here we will send a mail to the student
            return Response({'success':'outpasss declined successfully'})
            
        except:
            return Response({'error':'outpass not valid'})
        
class OutpassStudentsOut(StaffViewPermission):
    def get(self,request,format=None):
        staff_id=request.user.staff_profile.id
        staff=staff_profile.objects.get(id=staff_id)
        staff_type=staff.role
        if staff_type=="swc":
            students=staff.students_swc.all()
        elif staff_type=="warden":
            students=staff.students_warden.all()
        else:
            students=staff.students_fa.all()
        students=students.values_list('id', flat=True)
        s1=[]
        print(students)
        entryexitdetails=entry_exit.objects.filter(entry_time__isnull=True).filter(outpass__isnull=False).filter(roll_no__in=students)
        entryexitdetails=EntryExitDetailsSerializer(entryexitdetails,many=True).data
        return Response(entryexitdetails)
    
class AddStudentsForOutpass(StaffViewPermission):
    def get(self,request,format=None):
        students=student_profile.objects.all()
        students=ManageStudentsSerializer(students,many=True).data
        return Response(students)
    def post(self,request,format=None):
        students_list=request.data.getlist('students')
        students=student_profile.objects.filter(id__in=students_list)
        staff_id=request.user.staff_profile.id
        staff=staff_profile.objects.get(id=staff_id)
        staff_type=staff.role
        if staff_type=="swc":
            staff.students_swc.add(*students)
        elif staff_type=="warden":
            staff.students_warden.add(*students)
        else:
            staff.students_fa.add(*students)
        return Response({'success':'students addded'})
    
class StudentsUnderStaff(StaffViewPermission):
    def get(self,request,format=None):
        staff_id=request.user.staff_profile.id
        staff=staff_profile.objects.get(id=staff_id)
        staff_type=staff.role
        if staff_type=="swc":
            students=staff.students_swc.all()
        elif staff_type=="warden":
            students=staff.students_warden.all()
        else:
            students=staff.students_fa.all()
        students=ManageStudentsSerializer(students,many=True).data
        return Response(students)
    def post(self,request,format=None):
        staff_id=request.user.staff_profile.id
        staff=staff_profile.objects.get(id=staff_id)
        id=request.data['id']
        student=student_profile.objects.get(id=id)
        staff_type=staff.role
        if staff_type=="swc":
            staff.students_swc.remove(student)
        elif staff_type=="warden":
            staff.students_warden.remove(student)
        else:
            staff.students_fa.remove(student)
        return Response({'success':'successfully removed'})

class SwitchStaffRole(APIView):
    def post(self,request,format=None):
        data=request.data['access']
        data=data.split(",")
        curr_role=request.user.staff_profile.role
        role_list=['swc','fa','warden']
        if (data[0] in role_list) and (data[1] in role_list): 
            if data[0]==curr_role:
                request.user.staff_profile.role=data[1]
            else:
                request.user.staff_profile.role=data[0]
            request.user.staff_profile.save()
            print(request.user.staff_profile.role)
            return Response({'success':'switched to'+str(request.user.staff_profile.role)})
        else:
            return Response({'error':'wrong roles'})