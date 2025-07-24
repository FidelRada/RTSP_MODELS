import base64
import cv2
import numpy as np
from .FaceRecognizer import FaceRecognizer

def base64_to_image(base64_string):
    """
    Convierte una cadena base64 (sin encabezado o con) en una imagen OpenCV (numpy array)
    """
    try:
        # Quitar el encabezado si está presente (ej: "data:image/jpeg;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decodificar base64
        img_data = base64.b64decode(base64_string)
        
        # Convertir a numpy array
        np_arr = np.frombuffer(img_data, np.uint8)
        
        # Decodificar como imagen
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("No se pudo decodificar la imagen. Formato incorrecto.")
        
        return image
    except Exception as e:
        raise ValueError(f"Error al procesar la imagen base64: {str(e)}")
    
def registrar(imagen_base64, nombre_usuario):
    """
    Registra un rostro a partir de una imagen en base64 y un nombre de usuario.
    """
    try:
        
        # Convertir base64 a imagen OpenCV
        imagen = base64_to_image(imagen_base64)
        
        # Inicializar el reconocedor de rostros
        recognizer = FaceRecognizer()
        
        # Registrar el rostro
        result = recognizer.register_face(imagen, nombre_usuario)
        
        return result
    

    except Exception as e:
        print({"status": "error", "message": str(e)})