import yaml
import time
from multiprocessing import Process
from utils.video import opncv
from utils.yolo import yolo
from utils.tracker import reid


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
    server_config = load_config('conf/server.yaml')

    base_ip = server_config['base_ip']
    base_port = server_config['base_port']

    yolo_anno_port = base_port
    yolo_crop_port = base_port + 1
    reid_track_port = base_port + 2

    yolo_config['anno_port'] = yolo_anno_port
    yolo_config['crop_port'] = yolo_crop_port
    yolo_config['ip'] = base_ip
    yolo_config['raw_ports'] = []
    reid_config['crop_port'] = yolo_crop_port
    reid_config['track_port'] = reid_track_port
    reid_config['ip'] = base_ip

    processes = []

    current_port = base_port + 3
    for stream in stream_config['streams']:
        stream['ip'] = base_ip
        stream['port'] = current_port
        yolo_config['raw_ports'].append(current_port)
        processes.append(start_process(opncv, (stream,)))
        current_port += 1

    processes.append(start_process(yolo, (yolo_config,)))
    processes.append(start_process(reid, (reid_config,)))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
