import base64
with open('/home/sanket/Downloads/silhouette/media/media/default.jpg','rb') as image1:
    image_data=image1.read()
    string1=base64.b64decode(image_data).decode('latin1')
    print(string1)
