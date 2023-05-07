import qrcode
name="sanketmishra"
user_name="21bcs100"
unique_string=f"{name}-{user_name}"
qr=qrcode.QRCode(
    version=1,
    box_size=10,
    border=7,
)
qr.add_data(unique_string)
qr.make(fit=True)
img=qr.make_image(fill_color="black",back_color="white")
img.save(f"{user_name}_qr.png")