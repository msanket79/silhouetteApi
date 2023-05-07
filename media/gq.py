import cv2

img = cv2.imread('/qr.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detector = cv2.QRCodeDetector()
data, bbox, _ = detector.detectAndDecode(gray)

if data:
    print("QR code data:", data)

if bbox is not None:
    for i in range(len(bbox)):
        cv2.line(img, tuple(bbox[i][0]), tuple(bbox[(i+1) % len(bbox)][0]), color=(255, 0, 255), thickness=2)

cv2.imshow('QR code', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
