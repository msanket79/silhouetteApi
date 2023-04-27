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



import facial_classification as fc
classifier=fc.classifier()

class SecurityProfileView(APIView):
    def get(self,request,format=None):
        user=request.user
        security=security_profile.objects.get(admin=user)
        security=SecurityProfileSerializer(security).data
        return Response(security)
class ListOfStudentsOutSerializer(APIView):
    def get(self,request,format=None):
        entryexitdetails=entry_exit.objects.filter(entry_time__isnull=True).filter(outpass__isnull=True)
        entryexitdetails=EntryExitDetailsSerializer(entryexitdetails,many=True).data
        return Response(entryexitdetails)

class direct_entry(APIView):
    def post(self,request,format=None):
        entry_type=request.data['entry_type']
        if entry_type=="manual":
            roll_no=request.POST["roll_no"].upper()
        elif entry_type=="face_accept":
            roll_no=request.POST["roll_no"].upper()
        else:
            images = request.data.getlist('img')
            labels=classifier.testSVM(images)
            if len(labels)!=0:
                print(labels[0])
                try:
                    student=student_profile.objects.get(roll_no=labels[0].upper())
                    student=ScannedStudentSerializer(student).data
                    return Response(student)
                except:
                    return Response({'error':'student not in db'})
            else:
                return Response({'error':'no student is recognised'})

        try:
            student=student_profile.objects.get(roll_no=roll_no)

            entryexit=entry_exit.objects.filter(roll_no=student).filter(entry_time__isnull=True).first()
            query=(Q(roll_no=student) & Q(approved=True) & ~Q(used = True) & Q(From=datetime.date.today()+datetime.timedelta(hours=5,minutes=30)))
            outpass=Outpass.objects.filter(query).first()



            # agar student bahar hai with outpass
            if outpass and  outpass.used==False:
                outpass.entry.entry_time=datetime.datetime.now()+datetime.timedelta(hours=5,minutes=30)
                
                outpass.entry.save()
                outpass.used=True
                outpass.save()
                return Response({'success':'outpass entry done'})
             # direct entry exit
            elif  (not outpass) or entryexit :
                
                print(roll_no)
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
            else:
                # outpass or direct entry page render
                return Response({'outpass':roll_no})
                # return render(request,'users/security_template/outpass_or_direct.html',{"roll_no":roll_no})
        except:
                return Response({'error':'not valid student'})

class direct_or_outpass(APIView):
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
            





