import yaml
import time
from multiprocessing import Process, Manager
from utils.video import process_stream
from utils.yolo import process_frame
from utils.tracker2 import reid_process

def main():
    with open('conf/video.yaml', 'r') as file:
        stream_config = yaml.safe_load(file)
    with open('conf/yolo.yaml', 'r') as f:
        yolo_config = yaml.safe_load(f)
    with Manager() as manager:
        shared_raw_frames = manager.dict()
        shared_annotated_frames = manager.dict()
        shared_cropped_frames = manager.dict()
        shared_tracking_frames = manager.dict()
        for stream in stream_config['streams']:
            shared_raw_frames[stream['name']] = manager.dict()
            shared_annotated_frames[stream['name']] = manager.dict()
            shared_cropped_frames[stream['name']] = manager.dict()
            shared_tracking_frames[stream['name']] = manager.dict()


        lock = manager.Lock()  # Create a lock
        processes = []

        # Start yolo process
        p = Process(target=process_frame, args=(yolo_config,
                                                shared_raw_frames,
                                                shared_annotated_frames,
                                                shared_cropped_frames,
                                                lock
                                                ))
        p.start()
        processes.append(p)

        # Start tracker process
        p = Process(target=reid_process, args=(yolo_config,
                                               shared_cropped_frames,
                                               shared_tracking_frames,
                                               lock
                                               ))
        p.start()
        processes.append(p)

        # Start stream handlers
        for stream in stream_config['streams']:
            p = Process(target=process_stream, args=(stream,
                                                     shared_raw_frames,
                                                     lock
                                                     ))
            p.start()
            processes.append(p)

        try:
            while True:
                time.sleep(1)
                for stream_name in shared_raw_frames.keys():
                    print(f"{stream_name} has {len(shared_cropped_frames[stream_name])} people in frame")
        except KeyboardInterrupt:
            pass
        finally:
            for p in processes:
                p.join()


if __name__ == "__main__":
    main()
