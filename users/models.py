from django.db import models
from django.contrib.auth.models import AbstractUser
from django.dispatch import receiver
from django.db.models.signals import post_save

# from outpass.models import staff_profile

# Create your models here.

data=[["LALIT YADAV","B.Tech.","Data Science & Artificial Intelligence","22BDS034","A+","VISHNU KUMAR","03-Apr-2004","8307515689","Jul-26","7027063474"],
["MUSHAM SAI TARUN","B.Tech.","Data Science & Artificial Intelligence","21BDS040","Musham Laxman","02-Aug-2003","9392976627","AB+","Jul-25","9390465531"],
["KAMMINANA MAURYA","B.Tech.","Computer Science & Engineering","21BCS050","KAMMINANA Chennakesavarao","09-Jun-2003","9346048616","AB+","Jul-25","6309556773"],
["DONGA SAIPAVAN","B.Tech.","Computer Science & Engineering","21BCS035","Donga Satyanarayana","27-May-2004","9866182266","B+","Jul-25","9849614221"],
["AKKALAGARI ABILASH RAJENDER","B.Tech.","Computer Science & Engineering","21BCS007","Akkalagari Rajender Ramchander","24-Feb-2003","8888915165","B+","Jul-25","9822631646"],
["SAHIL KIRTI","B.Tech.","Computer Science & Engineering","21BCS097","Pradeep Ku. Patel","17-Nov-2002","7225965036","O+","Jul-25","8770523840"],
["MAKAM SUJITH","B.Tech.","Computer Science & Engineering","21BCS061","Makam Shankar Kiran","21-Jan-2004","8555042700","B+","Jul-25","9052573925"],
["YENNAPU PRANEETH","B.Tech.","Computer Science & Engineering","21BCS137","Yennapu Venkataratnam","09-Feb-2004","9133102513","O+","Jul-25","9246582919"],
["KUNGHADKAR PRATIK AWADHUT","B.Tech.","Computer Science & Engineering","21BCS058","Kunghadkar Awadhut","05-Jun-2003","9860839521","O+","Jul-25","9975993554"],
["ROUNAK AGRAWAL","B.Tech.","Electronics & Communication Engineering","21BEC039","Bijaykumar Agrawal","16-Feb-2003","9438038953","A+","Jul-25","9437036737"],
["SHIVAM KUMAR","B.Tech.","Computer Science & Engineering","21BCS112","Harendra singh","21-Jul-2002","6206641232","B+","Jul-25","9631326479"],
["SATHWIK NARKEDIMILLI","B.Tech.","Computer Science & Engineering","21BCS103","Narasimha Rao Narkedimilli Venkata Kanaka","17-Jul-2003","9845526132","A+","Jul-25","9867640114"],
["RAHUL VERMA","B.Tech.","Computer Science & Engineering","21BCS089","Om Prakash","29-Mar-2003","9110765860","B+","Jul-25","8328247495"],
["VISHWATEJA PEDDAKAPU","B.Tech.","Computer Science & Engineering","21BCS135","Satyanaryana Peddakapu","21-Oct-2002","7981247761","B+","Jul-25","9849086340"],
["SURYANSH RAJ","B.Tech.","Computer Science & Engineering","21BCS124","Rajeshkumar Singh","10-Oct-2002","8298832175","A+","Jul-25","7903320317"],
["MANDAVA KASHYAP SAI","B.Tech.","Computer Science & Engineering","21BCS063","Mandava Harikrishna","16-Oct-2003","8125607196","AB+","Jul-25","9908235823"],
["SHAIK SADIKH","B.Tech.","Data Science & Artificial Intelligence","21BDS059","Shaik Yesdan Basha","01-Sep-2003","9441553908","O+","Jul-25","9908163977"],
["VISHAL KUMAR","B.Tech.","Computer Science & Engineering","21BCS133","Santosh Kumar","27-Aug-2003","7490894360","O+","Jul-25","7600998651"],
["AJAY BHAKAR","B.Tech.","Computer Science & Engineering","21BCS006","Heera Ram","25-Dec-2002","7878340983","O+","Jul-25","8094052665"],
["PATHAK AYUSH DILIP","B.Tech.","Data Science & Artificial Intelligence","22BDS044","O+","PATHAK DILIP","05-Mar-2003","9316568042","Jul-26","9662400965"],
["DEVARSHI DUBEY","B.Tech.","Computer Science & Engineering","21BCS031","Anil Kumar Dubey","25-May-2003","9131499829","AB+","Jul-25","6266840013"],
["ROHAN CHIDRI","B.Tech.","Electronics & Communication Engineering","21BEC038","Rajkumar Chidri","10-Jul-2003","7204107880","A+","Jul-25","9448513958"],
["ASHISH JOY JONNAKUTI","B.Tech.","Computer Science & Engineering","21BCS016","Aneel VB Babu J","24-Jan-2004","9449648861","B+","Jul-25","9900240884"],
["SIRALA UDAY TEJA","B.Tech.","Computer Science & Engineering","21BCS121","Sirala Venkatasubbaiah","09-Jul-2003","9392741336","B+","Jul-25","9440306422"],
["MANCHALA SRINU","B.Tech.","Computer Science & Engineering","21BCS062","Manchala Thirupathaiah","27-Jul-2003","6304110817","O+","Jul-25","9392507833"],
["KARAN RAGHURAM NAIK","B.Tech.","Computer Science & Engineering","21BCS051","Raghuram L Naik","05-Mar-2003","7899999884","O+","Jul-25","9986424111"],
["DODDI BHUVANESH","B.Tech.","Data Science & Artificial Intelligence","21BDS018","D B G Tilak","26-Feb-2003","7780721858","B+","Jul-25","9866628686"],
["VIKASH TOMAR","B.Tech.","Data Science & Artificial Intelligence","22BDS063","B+","RAJKUMAR SINGH","07-Jul-2004","7340182247","Jul-26","8094137783"],
["HITESH SHARMA","B.Tech.","Data Science & Artificial Intelligence","22BDS028","A+","RAJVEER SHARMA","25-Aug-2004","8639601105","Jul-26","8919915726"],
["NIMMALA SUDARSHANA LAKSHMI VISWANADH","B.Tech.","Electronics & Communication Engineering","21BEC027","Nimmala Srinivasarao","16-May-2004","6300778837","A+","Jul-25","9440413609"],
["KRISHNA JAISWAL","B.Tech.","Computer Science & Engineering","21BCS056","Anil Jaiswal","24-Aug-2003","8840413100","B+","Jul-25","8400299331"],
["RAGHAVA GATADI","B.Tech.","Computer Science & Engineering","21BCS088","Gatadi Sreekanth","29-Apr-2004","9441190131","B+","Jul-25","9440090131"],
["MANISH KAUSHIK","B.Tech.","Data Science & Artificial Intelligence","22BDS037","B+","JAI PARKASH","30-Sep-2005","9350145621","Jul-26","7027517793"],
["HARSH MALIK","B.Tech.","Electronics & Communication Engineering","22BEC017","O+","HARENDRA MALIK","26-Jan-2003","6399944243","Jul-26","9598361230"],
["VINEET SANDEEP SEN","B.Tech.","Computer Science & Engineering","21BCS132","Sandeep Sen","09-Oct-2002","8600997083","O+","Jul-25","9320007083"],
["SHAIK ZUHAIR HASAN","B.Tech.","Data Science & Artificial Intelligence","21BDS060","Shaik Ibrahim","04-Jul-2004","9494292277","O+","Jul-25","9441937391"],
["PRATISH KUMAR DWIVEDI","B.Tech.","Electronics & Communication Engineering","22BEC034","O+","NARAYAN PRASAD DWIVEDI","10-Jul-2003","7480082988","Jul-26","7210769097"],
["KSHITIZ SACHAN","B.Tech.","Computer Science & Engineering","21BCS057","Gyan Deo Sachan","16-Oct-2003","9559790932","O+","Jul-25","6394409162"],
["NITHISH CHOUTI","B.Tech.","Computer Science & Engineering","21BCS074","Ramesh CHOUTI","02-May-2003","9493865924","O+","Jul-25","7995466264"],
["DUNGAVATH SANTHOSH NAIK","B.Tech.","Computer Science & Engineering","21BCS036","Dungavath Ramu Naik","06-Apr-2003","6300293214","O+","Jul-25","8106994833"],
["AMBALLA VENKATA SRI RAM","B.Tech.","Computer Science & Engineering","21BCS008","Amballa Venkata Ram Rao","19-Sep-2003","7075023539","B-","Jul-25","9849211471"]]
class customUser(AbstractUser):
    email = models.EmailField(unique=True)
    user_type_data=((1,'admin'),(2,'security'),(3,'student'),(4,'staff'),)
    user_type=models.CharField(default=1,choices=user_type_data,max_length=100)


class student_profile(models.Model):
    id=models.AutoField(primary_key=True)
    name=models.CharField(max_length=200)
    roll_no=models.CharField(unique=True,max_length=100)
    father=models.CharField(max_length=200)
    phone_no=models.CharField(max_length=100)
    emergency_phone_no=models.CharField(max_length=100)
    gender=models.CharField(max_length=100,default="male")
    profile_pic=models.FileField(max_length=400,default="media/default.jpg")
    ban=models.BooleanField(null=True,blank=True,default=False)
    objects=models.Manager()
    admin=models.OneToOneField(customUser,on_delete=models.CASCADE)
class appeal_unban(models.Model):
    reason=models.TextField(max_length=1000)
    student_id=models.OneToOneField(student_profile,null=True,blank=True,on_delete=models.CASCADE)

class security_profile(models.Model):
    id=models.AutoField(primary_key=True)
    phone_no=models.CharField(max_length=100)
    name=models.CharField(max_length=200)
    profile_pic=models.FileField(max_length=400,default="media/default.jpg")
    head=models.BooleanField(default=False,blank=True)
    admin=models.OneToOneField(customUser,on_delete=models.CASCADE)
    objects=models.Manager()

class admin_profile(models.Model):
    id=models.AutoField(primary_key=True)
    name=models.CharField(max_length=200)
    phone_no=models.CharField(max_length=200,null=True,blank=True)
    profile_pic=models.FileField(max_length=400,default="media/default.jpg")
    admin=models.OneToOneField(customUser,on_delete=models.CASCADE)
    objects=models.Manager()

class entry_exit(models.Model):
    id=models.AutoField(primary_key=True)
    roll_no=models.ForeignKey(student_profile,on_delete=models.CASCADE,null=True,blank=True)
    exit_time=models.DateTimeField(auto_now_add=True)
    entry_time=models.DateTimeField(null=True,blank=True)
    objects=models.Manager()




