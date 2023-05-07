from Crypto.Cipher import AES
import base64

def encrypt(message, secret_key):
    """
    Encrypts the given message using the CryptoJS AES encryption algorithm.
    """
    # Create an AES cipher object and set it to encryption mode
    key = secret_key.encode('utf-8')
    iv = b'0000000000000000'  # use a fixed initialization vector
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Add padding to the message so that its length is a multiple of 16
    length = 16 - (len(message) % 16)
    message += chr(length) * length

    # Encrypt the message and return it as a base64-encoded string
    ciphertext = cipher.encrypt(message.encode('utf-8'))
    return base64.b64encode(ciphertext).decode('utf-8')

def decrypt(ciphertext, secret_key):
    """
    Decrypts the given ciphertext using the CryptoJS AES encryption algorithm.
    """
    # Create an AES cipher object and set it to decryption mode
    key = secret_key.encode('utf-8')
    iv = b'0000000000000000'  # use a fixed initialization vector
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Decrypt the ciphertext and remove the padding
    ciphertext = base64.b64decode(ciphertext)
    plaintext = cipher.decrypt(ciphertext).rstrip(b"\0")

    # Convert the decrypted message to a string and return it
    return plaintext.decode('utf-8')
