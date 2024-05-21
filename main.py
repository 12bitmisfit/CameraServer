import yaml
import time
from multiprocessing import Process, Manager
from utils.video import opncv
from utils.yolo import yolo
from utils.tracker2 import reid


def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def start_process(target, args):
    process = Process(target=target, args=args)
    process.start()
    return process


def main():
    stream_config = load_config('conf/video.yaml')
    yolo_config = load_config('conf/yolo.yaml')
    reid_config = load_config('conf/reid.yaml')

    with Manager() as manager:
        shared_raw = manager.dict()
        shared_anno = manager.dict()
        shared_crop = manager.dict()
        shared_track = manager.dict()

        for stream in stream_config['streams']:
            shared_raw[stream['name']] = manager.dict()
            shared_anno[stream['name']] = manager.dict()
            shared_crop[stream['name']] = manager.dict()
            shared_track[stream['name']] = manager.dict()

        lock = manager.Lock()
        processes = []

        for stream in stream_config['streams']:
            processes.append(start_process(opncv, (stream, shared_raw, lock)))

        processes.append(start_process(yolo, (yolo_config, shared_raw, shared_anno, shared_crop, lock)))
        processes.append(start_process(reid, (reid_config, shared_crop, shared_track, lock)))

        try:
            while True:
                time.sleep(1)
                for stream_name in shared_raw.keys():
                    print(f"{stream_name} has {len(shared_crop[stream_name])} people in frame")
        except KeyboardInterrupt:
            pass
        finally:
            for p in processes:
                p.join()


if __name__ == "__main__":
    main()
