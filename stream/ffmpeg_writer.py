# stream/ffmpeg_writer.py

import subprocess
import os
from config import OUTPUT_DIR, RESOLUTION, FPS

def start_ffmpeg(output_path):
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    width, height = RESOLUTION
    print(output_path + '************************************')

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "5",
        "-hls_flags", "delete_segments",
        os.path.join(f"{output_path}", "stream.m3u8")
        #hls_output/cam0
    ]
    
    try :
        subproceso = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        print("la puta que te pario maldito ffmpeg")
        return subproceso
    except :
        print("paso algo")

