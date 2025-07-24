import base64
import numpy as np
import cv2
from PIL import Image
import pytesseract
import json
from .recortarCarnet import recortar, recortar_en_memoria
from .TextExtract import extractTextFromImage
from .NuExtrat_tiny import predict_NuExtract

#def procesar_carnet_base64(imagen_base64: str):
#    # 1. Convertir imagen base64 a np.ndarray (OpenCV)
#    imagen_bytes = base64.b64decode(imagen_base64)
#    imagen_array = np.frombuffer(imagen_bytes, np.uint8)
#    imagen_cv2 = cv2.imdecode(imagen_array, cv2.IMREAD_COLOR)
#
#    #if imagen_cv2 is None:
#    #    raise ValueError("No se pudo decodificar la imagen.")
#
#    # 2. Recorte
#    carnet_recortado = recortar_en_memoria(imagen_cv2)
#
#    # 3. OCR
#    imagen_pil = Image.fromarray(cv2.cvtColor(carnet_recortado, cv2.COLOR_BGR2RGB))
#    texto = pytesseract.image_to_string(imagen_pil, lang='spa')
#
#    # 4. Predicción con plantilla
#    template = '''{
#        "Nombre": "",
#        "Apellido": ""
#    }'''
#    result = predict_NuExtract([texto], template)
#    return json.loads(result[0])

def procesar_carnet_base64(imagen_base64: str):
    # 1. Convertir imagen base64 a np.ndarray (OpenCV)
    try:
        imagen_base64 = imagen_base64.split(',')[-1]  # Por si viene con el prefijo "data:image/jpeg;base64,"
        imagen_bytes = base64.b64decode(imagen_base64)
        imagen_array = np.frombuffer(imagen_bytes, np.uint8)
        imagen_cv2 = cv2.imdecode(imagen_array, cv2.IMREAD_COLOR)

        if imagen_cv2 is None or imagen_cv2.size == 0:
            raise ValueError("La imagen está vacía o no se pudo decodificar correctamente.")

        # 2. Recorte
        carnet_recortado = recortar_en_memoria(imagen_cv2)

        # 3. OCR
        imagen_pil = Image.fromarray(cv2.cvtColor(carnet_recortado, cv2.COLOR_BGR2RGB))
        texto = pytesseract.image_to_string(imagen_pil, lang='spa')

        # 4. Predicción con plantilla
        template = '''{
            "Nombre": "",
            "Apellido": ""
        }'''
        result = predict_NuExtract([texto], template)
        return json.loads(result[0])

    except Exception as e:
        raise RuntimeError(f"Error procesando el carnet: {e}")