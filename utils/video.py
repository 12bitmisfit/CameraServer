import cv2
import time
import os


def process_stream(stream_config, shared_raw_frames, lock):
    cap = cv2.VideoCapture(stream_config['url'])
    if not cap.isOpened():
        print(f"Failed to open video stream: {stream_config['url']}")
        return

    output_fps = stream_config['output_fps']
    output_frame_period = 1.0 / output_fps
    shared_frame_period = 1
    next_output_time = time.time() + output_frame_period
    next_shared_time = time.time() + shared_frame_period

    total_frames = int(output_fps * stream_config['length'])

    while True:
        segment_frames = 0
        out = None

        while segment_frames < total_frames:
            ret, frame = cap.read()
            if not ret:
                print("No frame read from the stream")
                continue  # Skip the frame processing if no frame is read

            current_time = time.time()
            if current_time >= next_output_time:
                if segment_frames % total_frames == 0 or out is None:
                    if out is not None:
                        out.release()
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    date_path = time.strftime("%Y/%m/%d/")
                    base_path = os.path.join(stream_config['output_dir'], stream_config['name'], date_path)
                    os.makedirs(base_path, exist_ok=True)
                    file_name = f"{stream_config['name']}_output_{timestamp.split('-')[1]}.avi"
                    out_path = os.path.join(base_path, file_name)
                    fourcc = cv2.VideoWriter.fourcc(*'MJPG')
                    out = cv2.VideoWriter(out_path, fourcc, output_fps, tuple(stream_config['output_resolution']))

                if stream_config['output_enabled']:
                    output_frame = cv2.resize(frame, tuple(stream_config['output_resolution']))
                    out.write(output_frame)

                next_output_time += output_frame_period
                segment_frames += 1

            if current_time >= next_shared_time:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with lock:
                        shared_raw_frames[stream_config['name']]['raw_frame'] = (rgb_frame, current_time)
                    next_shared_time += shared_frame_period
                except Exception as e:
                    print(f"A video error occurred: {e}")
                    continue

        if out:
            out.release()
        cap.release()  # Ensure the capture is released outside the loop


