from django.core.mail import send_mail
from silhouette.settings import EMAIL_HOST_USER
from outpass.models import staff_profile
def outpass_pending_send_nightly_email():
    # saare staffs nikaal liye
    staffs=staff_profile.objects.all()
    # now itertating over each staff
    for staff in staffs:
        pass
    subject = 'Your subject here'
    message = 'Your message here'
    recipient_list = ['recipient_email@example.com']
    send_mail(subject, message, EMAIL_HOST_USER, recipient_list, fail_silently=False)