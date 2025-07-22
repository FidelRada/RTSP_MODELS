# server/web_server.py

import os
from flask import Flask, send_from_directory, render_template, request, jsonify
from flask_cors import CORS
from config import OUTPUT_DIR
import threading
from stream.camera import stream_camera_ffmpeg
from stream.stream_with_recognition import stream_camera_with_face

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