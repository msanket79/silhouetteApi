from .models import entry_exit,student_profile,security_profile
from outpass.models import Outpass
from django.db.models import Q
import datetime

from rest_framework.views import APIView
from rest_framework import permissions
from rest_framework.response import Response
from django.contrib import auth
from .serializers import SecurityProfileSerializer,EntryExitDetailsSerializer,ScannedStudentSerializer

from rest_framework.decorators import api_view
from django.urls import reverse
import requests

from django.conf import settings

media_root = settings.MEDIA_ROOT

from time import time
import base64
import os

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from .permissions import  IsSecurity
from .models import appeal_unban
class SecurityViewPermission(APIView):
    permission_classes=[IsSecurity]

def save_base64_image(base64_string, folder):
    # Decode base64 data
    format, imgstr = base64_string.split(';base64,')
    ext = format.split('/')[-1]
    data = ContentFile(base64.b64decode(imgstr))

    # Generate unique filename
    filename = os.path.join(folder, f"myphoto.{ext}")
    if default_storage.exists(filename):
        print('it exists')
        default_storage.delete(filename)
    # Save the file in Django's media folder
    path = default_storage.save(filename, data)

    # Return the file path
    return default_storage.url(path)+f'?t={int(time())}'


import facial_classification as fc
classifier=fc.classifier()



# this is a security profile 
class SecurityProfileView(SecurityViewPermission):
    def get(self,request,format=None):
        user=request.user
        security=security_profile.objects.get(admin=user)
        security=SecurityProfileSerializer(security).data
        return Response(security)
    
#this is the listofstudents out
class ListOfStudentsOutSerializer(SecurityViewPermission):
    def get(self,request,format=None):
        entryexitdetails=entry_exit.objects.filter(entry_time__isnull=True).filter(outpass__isnull=True)
        entryexitdetails=EntryExitDetailsSerializer(entryexitdetails,many=True).data
        return Response(entryexitdetails)


# this is used to do face entry and manual entry with differnt post parameters

# if entry_type==manual then it will used for manual entry
# if entry_type==face_accept then we have showed the details of the student after successfull recognition and then user is clicking sumbit
# else it will process recieve the 10 images and will process it 
class direct_entry(SecurityViewPermission):
    def post(self,request,format=None):
        entry_type=request.data['entry_type']
        # when we want to do entry using the roll no
        if entry_type=="manual":
            roll_no=request.POST["roll_no"].upper()
            try:
                student=student_profile.objects.get(roll_no=roll_no)
                student=ScannedStudentSerializer(student).data
                return Response(student)
            except:
                return Response({'error':'student not in db'})
        # this is the page when face is scanned and i post a student details to frontend and they click on submit
        elif entry_type=="face_accept":
            roll_no=request.POST["roll_no"].upper()
        # this is when the security clicks on the face entry and 10 photos are sent
        else:
            images = request.data.getlist('img')
            labels=classifier.testSVM(images)
            if len(labels)!=0:
                print(labels[0])
                try:
                    student=student_profile.objects.get(roll_no=labels[0].upper())
                    student=ScannedStudentSerializer(student).data
                    response=student
                    response['image']=save_base64_image(images[2],media_root)
                    print(response)
                    return Response(student)
                except Exception as e:
                    
                    print(e)
                    return Response({'error':'student not in db'})
            else:
                return Response({'error':'no student is recognised'})

        try:
            student=student_profile.objects.get(roll_no=roll_no)
            entryexit=entry_exit.objects.filter(roll_no=student).filter(entry_time__isnull=True).first()
            query=(Q(roll_no=student) & Q(approved=True) & ~Q(used = True) & Q(From=datetime.datetime.now()))
            outpass=Outpass.objects.filter(query).first()


            # agar student bahar hai with outpass
            if outpass and  outpass.used==False:
                outpass.entry.entry_time=datetime.datetime.now()
                
                outpass.entry.save()
                outpass.used=True
                outpass.save()
                return Response({'success':'outpass entry done'})
             # direct entry exit
            elif  (not outpass) or entryexit :

                try:
                    student=student_profile.objects.get(roll_no=roll_no)
                    entryexit=entry_exit.objects.filter(roll_no=student).filter(entry_time__isnull=True).first()
                    if not entryexit: # if entry done

                        if not student.ban:
                            entryexit1=entry_exit.objects.create(roll_no=student)
                            entryexit1.exit_time=entryexit1.exit_time
                            entryexit1.save()
                            return Response({'success':'exit done'})
                        else:
                            return Response({'error':'student is banned'})
                    else:
                        now=datetime.datetime.now()
                        entryexit.entry_time=now
                        # here we have to ban the students

                        if now.time() > datetime.time(hour=21, minute=30):
                            student.ban=True
                            req=appeal_unban.objects.create(student_id=student,cause="Student was late and came to campus at "+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) )
                            req.save()
                        entryexit.save()
                        return Response({'success':'entry done'})
                except Exception as e:
                    print(e)
                    return Response({'error':'enter a valid roll no'})
            else:
                # outpass or direct entry page render
                return Response({'outpass':roll_no})
                # return render(request,'users/security_template/outpass_or_direct.html',{"roll_no":roll_no})
        except:
                return Response({'error':'not valid student'})



# this is the page will appear when the student want to exit adn he has outpass in his name and it will ask direct or outpass
class direct_or_outpass(SecurityViewPermission):
    def post(self,request,format=None):
        roll_no=request.data["roll_no"].upper()
        exit_type=request.data['exit_type']
        if exit_type=="outpass":
            student=student_profile.objects.get(roll_no=roll_no)
            print(student)
            query=(Q(roll_no=student) & Q(approved=True) & ~Q(used=True))
            outpasses=Outpass.objects.filter(query).first()
            print(outpasses)
            entryexit1=entry_exit.objects.create(roll_no=student)
            entryexit1.exit_time=entryexit1.exit_time+datetime.timedelta(hours=5,minutes=30)

            outpasses.entry=entryexit1
            outpasses.used=False
            outpasses.save()
            return Response({'success':'outpass exit done'})
        else:
            try:
                student=student_profile.objects.get(roll_no=roll_no)
                entryexit=entry_exit.objects.filter(roll_no=student).filter(entry_time__isnull=True).first()
                if not entryexit: # if entry done
                    if not student.ban:
                        entryexit1=entry_exit.objects.create(roll_no=student)
                        entryexit1.exit_time=entryexit1.exit_time+datetime.timedelta(hours=5,minutes=30)
                        entryexit1.save()
                        return Response({'success':'exit done'})
                    else:
                        return Response({'error':'student is banned'})
                else:
                    entryexit.entry_time=datetime.datetime.now()+datetime.timedelta(hours=5,minutes=30)
                    print(entryexit)
                    entryexit.save()
                    return Response({'success':'entry done'})
            except:
                return Response({'error':'enter a valid roll no'})
            





