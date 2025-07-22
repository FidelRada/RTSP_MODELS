import os
import cv2
import numpy as np
import face_recognition

class FaceRecognizer:
    def __init__(self, database_path="database", threshold=0.6, resize=4):
        self.threshold = threshold
        self.database_path = database_path
        self.database = self._load_database(database_path)
        self.resize = resize
        self.resize_factor = 1 / resize
        
    def _load_database(self, path):
        print(path)
        """Carga todos los encodings de la base de datos"""
        os.makedirs(path, exist_ok=True)
        database = {}
        for file in os.listdir(path):
            name = os.path.splitext(file)[0]
            encoding = np.load(os.path.join(path, file))
            database[name] = encoding
        return database
    
    def process_frame(self, frame):
        """Procesa un frame y devuelve los resultados"""
        # Reducir tamaño para mejor performance
        small_frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        
        # Detectar rostros
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Escalar coordenadas al tamaño original
            top *= self.resize
            right *= self.resize
            bottom *= self.resize
            left *= self.resize
            
            # Buscar coincidencias
            name = "Desconocido"
            confidence = 1.0
            for db_name, db_encoding in self.database.items():
                distance = face_recognition.face_distance([db_encoding], face_encoding)[0]
                if distance < self.threshold and distance < confidence:
                    name = db_name
                    confidence = distance
            
            results.append({
                "nombre": name,
                "reconocido": name != "Desconocido",
                "confianza": float(confidence),
                "coordenadas": (left, top, right, bottom)  # (x1, y1, x2, y2)
            })
        
        return results

    def register_face(self, frame, user_name):
        
        """Registra un nuevo rostro desde un frame"""
        # Verificar si el usuario ya existe
        if user_name in self.database:
            return {"status": "error", "message": "El usuario ya existe"}
        
        # Detectar rostros en el frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_encodings = face_recognition.face_encodings(small_frame)
        
        if not face_encodings:
            return {"status": "error", "message": "No se detectaron rostros"}
        
        if len(face_encodings) > 1:
            return {"status": "error", "message": "Se detectaron mas de un rostro"}
        
        # Tomar el primer rostro detectado
        encoding = face_encodings[0]
        
        # Guardar en la base de datos
        np.save(os.path.join(self.database_path, f"{user_name}.npy"), encoding)
        
        # Actualizar base de datos en memoria
        self.database[user_name] = encoding
        
        return {"status": "success", "message": f"Usuario {user_name} registrado"}

# ----------------------------
# Ejemplo de uso con video
# ----------------------------

def _run_face_recognition(video_source=0):
    recognizer = FaceRecognizer(threshold=0.6)
    
    # Para usar con webcam (0) o archivo de video
    cap = cv2.VideoCapture(video_source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Procesar frame
        resultados = recognizer.process_frame(frame)
        
        # Dibujar resultados (opcional)
        for result in resultados:
            left, top, right, bottom = result["coordenadas"]
            label = f"{result['nombre']} ({result['confianza']:.2f})"
            
            if result['nombre'] != "Desconocido":
                color = (0, 255, 0) # azul
            else:
                color = (0, 0, 255) # rojo
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Mostrar frame (opcional)
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Ejecutar con webcam
#_run_face_recognition()

# Ejecutar con archivo de video
#_run_face_recognition("video.mp4")

def _run_face_recognition(video_source=0):
    recognizer = FaceRecognizer(threshold=0.6)
    cap = cv2.VideoCapture(video_source)
    register_mode = False
    user_name = ""

    #print("#################3",recognizer.database,"###############")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1)
        
        # Tecla 'r' para iniciar registro
        if key == ord('r'):
            register_mode = True
            user_name = input("Ingrese el nombre del usuario: ")
        
        # Procesamiento normal
        resultados = recognizer.process_frame(frame)
        
        # Modo registro
        if register_mode:
            if user_name:
                result = recognizer.register_face(frame, user_name)
                print(result["message"])
                register_mode = False
                user_name = ""
            cv2.putText(frame, "Mire a la camara", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Dibujar resultados
        for result in resultados:
            left, top, right, bottom = result["coordenadas"]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, result["nombre"], (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ejecutar con webcam
#_run_face_recognition(0)
