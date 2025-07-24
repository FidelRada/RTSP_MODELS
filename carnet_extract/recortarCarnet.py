import cv2
import numpy as np

# 1. Cargar la imagen original
#def recortar(imagen_path):
#    imagen = cv2.imread(imagen_path)
#
#    if imagen is None:
#        print("No se pudo cargar la imagen. Verifica la ruta.")
#        exit()
#
#    # 2. Convertir a escala de grises
#    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
#
#    # 3. Aplicar suavizado para reducir ruido
#    suave = cv2.GaussianBlur(gris, (5, 5), 0)
#
#    # 4. Detección de bordes con Canny
#    bordes = cv2.Canny(suave, 50, 150)
#
#    # 5. Buscar contornos
#    contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#    # 6. Filtrar los contornos buscando rectángulos (forma del carnet)
#    carnet_contorno = None
#    max_area = 0
#
#    for contorno in contornos:
#        perimetro = cv2.arcLength(contorno, True)
#        aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
#
#        # Buscamos un polígono cerrado de 4 lados (rectángulo)
#        if len(aprox) == 4:
#            area = cv2.contourArea(contorno)
#            if area > max_area:
#                max_area = area
#                carnet_contorno = aprox
#
#    if carnet_contorno is None:
#        print("No se encontró un contorno rectangular que parezca un carnet.")
#    else:
#        # 7. Dibujar el contorno encontrado (opcional)
#        cv2.drawContours(imagen, [carnet_contorno], -1, (0, 255, 0), 2)
#
#        # 8. Obtener coordenadas del rectángulo que encierra el carnet
#        x, y, w, h = cv2.boundingRect(carnet_contorno)
#
#        # 9. Recortar la región del carnet
#        carnet_recortado = imagen[y:y+h, x:x+w]
#        
#        carnet_recortado = cv2.resize(carnet_recortado, (1032, 666), interpolation=cv2.INTER_AREA)
#
#        # 10. Mostrar las imágenes (opcional)
#        cv2.imshow("Imagen Original", imagen)
#        cv2.imshow("Carnet Recortado", carnet_recortado)
#
#        # 11. Guardar la imagen recortada
#        cv2.imwrite("carnet_recortado.jpg", carnet_recortado)
#        print("✅ Carnet recortado guardado como 'carnet_recortado.jpg'")
#
#        # Esperar a que el usuario cierre las ventanas
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#    return "carnet_recortado.jpg"

def recortar(imagen_path):
    imagen = cv2.imread(imagen_path)

    if imagen is None:
        print("No se pudo cargar la imagen. Verifica la ruta.")
        return None

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(suave, 50, 150)

    contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    carnet_contorno = None
    max_area = 0

    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        if len(aprox) == 4:
            area = cv2.contourArea(contorno)
            if area > max_area:
                max_area = area
                carnet_contorno = aprox

    if carnet_contorno is None:
        print("No se encontró un contorno rectangular que parezca un carnet.")
        return None

    x, y, w, h = cv2.boundingRect(carnet_contorno)
    carnet_recortado = imagen[y:y+h, x:x+w]
    carnet_recortado = cv2.resize(carnet_recortado, (1032, 666), interpolation=cv2.INTER_AREA)

    salida_path = f"recortado_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(salida_path, carnet_recortado)
    return salida_path

def recortar_en_memoria(imagen: np.ndarray):
    if imagen is None or imagen.size == 0:
        raise ValueError("La imagen proporcionada está vacía o no es válida.")

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(suave, 50, 150)
    contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    carnet_contorno = None
    max_area = 0
    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        if len(aprox) == 4:
            area = cv2.contourArea(contorno)
            if area > max_area:
                max_area = area
                carnet_contorno = aprox

    if carnet_contorno is None:
        raise ValueError("No se encontró un contorno rectangular que parezca un carnet.")

    x, y, w, h = cv2.boundingRect(carnet_contorno)
    carnet_recortado = imagen[y:y + h, x:x + w]
    carnet_recortado = cv2.resize(carnet_recortado, (1032, 666), interpolation=cv2.INTER_AREA)
    return carnet_recortado


