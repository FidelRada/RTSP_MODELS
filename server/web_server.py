# server/web_server.py

import os
from flask import Flask, send_from_directory, render_template, request, jsonify
from flask_cors import CORS
from config import OUTPUT_DIR
import threading
import base64
import numpy as np
import cv2
from stream.camera import stream_camera_ffmpeg
from stream.stream_with_recognition import stream_camera_with_face
from stream.stream_with_parking import stream_camera_with_parking
from parking_detector.segment import Segment
from carnet_extract.extractor import procesar_carnet_base64
from my_face_recognition.Registro import registrar

app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route('/')
def index():
    cam_id = request.args.get("cam", default="0")
    return render_template("index.html", cam_id=cam_id)

# @app.route('/<path:filename>')
# def stream_files(filename):
#     return send_from_directory(f'../{OUTPUT_DIR}', filename)

@app.route('/<string:cam_name>/<path:filename>')
def stream_files(cam_name, filename):
    #path = os.path.abspath(os.path.join(f'../{OUTPUT_DIR}', f"cam{cam_id}"))
    print(f"../{OUTPUT_DIR}/{cam_name}")
    return send_from_directory(f"../{OUTPUT_DIR}/{cam_name}", filename)

@app.route('/start_stream', methods=['POST'])
def start_stream():
    data = request.get_json()
    rtsp_url = data.get('rtsp_url')
    folder_name = data.get('folder_name')
    if not rtsp_url or not folder_name:
        return jsonify({"error": "Faltan parámetros: rtsp_url y folder_name son requeridos"}), 400
    output_path = os.path.join(OUTPUT_DIR, folder_name)
    # Lanzar la transmisión en un hilo
    t = threading.Thread(target=stream_camera_ffmpeg, args=(rtsp_url, output_path))
    t.daemon = True
    t.start()
    m3u8_url = f"/{folder_name}/stream.m3u8"
    return jsonify({
        "message": "Transmisión registrada y lanzada correctamente",
        "m3u8_url": m3u8_url
    }), 200

@app.route('/start_stream_face', methods=['POST'])
def start_stream_face():
    data = request.get_json()
    rtsp_url = data.get('rtsp_url')
    folder_name = data.get('folder_name')
    if not rtsp_url or not folder_name:
        return jsonify({"error": "Faltan parámetros: rtsp_url y folder_name son requeridos"}), 400
    
    output_path = os.path.join(OUTPUT_DIR, folder_name)
    
    # Lanzar la transmisión y reconocimiento en un hilo
    thread = threading.Thread(target=stream_camera_with_face, args=(rtsp_url, output_path))
    thread.daemon = True
    thread.start()
    
    m3u8_url = f"/{folder_name}/stream.m3u8"
    return jsonify({
        "message": "Transmisión y reconocimiento facial registrados y lanzados correctamente",
        "m3u8_url": m3u8_url
    }), 200

@app.route('/start_stream_parking', methods=['POST'])
def start_stream_parking_route():
    data = request.get_json()
    
    rtsp_url = data.get('rtsp_url')
    folder_name = data.get('folder_name')
    parking_segments_json = data.get('parking_segments')
    special_segments_json = data.get('special_segments')

    # Validación de parámetros
    if not all([rtsp_url, folder_name, parking_segments_json, special_segments_json]):
        return jsonify({
            "error": "Faltan parámetros: rtsp_url, folder_name, parking_segments y special_segments son requeridos"
        }), 400

    # Ruta de salida
    output_path = os.path.join(OUTPUT_DIR, folder_name)

    # Convertir JSON a objetos Segment con validación
    try:
        parking_segments = [
            Segment.from_dict(seg) for seg in parking_segments_json
        ]
        special_segments = [
            Segment.from_dict(seg) for seg in special_segments_json
        ]
    except ValueError as e:
        return jsonify({
            "error": f"Datos de segmento inválidos: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            "error": f"Error inesperado al procesar segmentos: {str(e)}"
        }), 400

    # Iniciar el hilo de procesamiento
    thread = threading.Thread(
        target=stream_camera_with_parking,
        args=(rtsp_url, output_path, parking_segments, special_segments)
    )
    thread.daemon = True
    thread.start()

    m3u8_url = f"/{folder_name}/stream.m3u8"
    return jsonify({
        "message": "Transmisión y detección de parqueo iniciadas correctamente",
        "m3u8_url": m3u8_url
    }), 200
    
@app.route("/procesar_carnet", methods=["POST"])
def procesar_carnet():
    try:
        data = request.get_json()
        imagen_base64 = data.get("imagen_base64")

        if not imagen_base64:
            return jsonify({
                "error": "Falta el campo 'imagen_base64'"
            }), 400

        resultado = procesar_carnet_base64(imagen_base64)

        return jsonify({
            "data": resultado,
            "codigoRes": 200,
            "mensaje": "Extracción exitosa del carnet"
        })

    except Exception as e:
        return jsonify({
            "mensaje": str(e),
            "typeError": "ProcesamientoError",
            "codigoError": 500
        }), 500

@app.route("/registrar_rostro", methods=["POST"])
def registrar_rostro():
    try:
        data = request.get_json()
        imagen_base64 = data.get("imagen_base64")
        nombre_usuario = data.get("nombre")
        respuesta = registrar(imagen_base64, nombre_usuario)
        return jsonify(respuesta), 200

    except Exception as e:
        return jsonify({
            "mensaje": str(e),
            "typeError": "ErrorServidor",
            "codigoError": 500
        }), 500