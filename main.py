import yaml
from multiprocessing import Process, Manager
from utils.video import process_stream

'''
shared_frames struct:
{
    'stream_name': {
                'raw_frame': tuple(cv2_image, time.time() of when it was updated)
                'annotated_frame': (cv2_image, copy of raw_frame time)
                'cropped_frames': [] # list of cv2_images cropped to bounding boxes
    }
}

'''


def main():
    with open('conf/video.yaml', 'r') as file:
        stream_config = yaml.safe_load(file)
    with open('conf/yolo.yaml', 'r') as f:
        yolo_config = yaml.safe_load(f)
    manager = Manager()
    shared_raw_frames = manager.dict()
    shared_annotated_frames = manager.dict()
    shared_cropped_frames = manager.dict()
    lock = manager.Lock()  # Create a lock
    processes = []

    # Start yolo process
    p = Process(target=process_stream, args=(yolo_config,
                                             shared_raw_frames,
                                             shared_annotated_frames,
                                             shared_cropped_frames,
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
            print(shared_raw_frames.keys())  # Display active streams
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
