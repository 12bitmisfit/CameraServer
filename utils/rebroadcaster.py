from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import threading
import uvicorn


def capture_frames(video_config, current_frame, frame_lock):
    video_capture = cv2.VideoCapture(video_config['url'])
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_config['url']}")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        with frame_lock:
            current_frame[0] = frame

    video_capture.release()


def generate_mjpeg(current_frame, frame_lock):
    while True:
        with frame_lock:
            if current_frame[0] is None:
                continue
            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', current_frame[0])
            if not ret:
                continue
        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + f"{len(jpeg)}".encode() + b'\r\n\r\n' + jpeg.tobytes() + b'\r\n')


def create_app(current_frame, frame_lock):
    app = FastAPI()

    @app.get('/video')
    async def video_feed():
        return StreamingResponse(generate_mjpeg(current_frame, frame_lock),
                                 media_type='multipart/x-mixed-replace; boundary=frame')

    return app


def rebroadcast(video_config, output_port):

    current_frame = [None]  # Shared frame among threads
    frame_lock = threading.Lock()

    capture_thread = threading.Thread(target=capture_frames,
                                      args=(video_config, current_frame, frame_lock),
                                      daemon=True
                                      )
    capture_thread.start()
    app = create_app(current_frame, frame_lock)
    uvicorn.run(app, host='localhost', port=output_port)
