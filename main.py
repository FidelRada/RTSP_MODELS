# main.py

import threading
from stream.camera import stream_camera_ffmpeg
from server.web_server import app
from config import CAMERA_IDS, OUTPUT_DIR, PORT

if __name__ == '__main__':
    for cam_id in CAMERA_IDS:
        output_path = f"{OUTPUT_DIR}/cam{cam_id}"
        t = threading.Thread(target=stream_camera_ffmpeg, args=(cam_id, output_path))
        t.daemon = True
        t.start()

    print(f"🌐 Servidor web en: http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT, threaded=True)
