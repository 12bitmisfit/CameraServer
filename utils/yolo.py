from ultralytics import YOLO
import time
import zmq
import numpy as np
import os


def yolo(yolo_config):
    os.environ['YOLO_VERBOSE'] = 'False'
    model = YOLO(yolo_config['model_path'], verbose=False)
    context = zmq.Context()

    # Connect to multiple raw frame publishers
    raw_sub_sockets = []
    for port in yolo_config['raw_ports']:
        raw_sub_socket = context.socket(zmq.SUB)
        raw_sub_socket.connect(f"tcp://{yolo_config['ip']}:{port}")
        raw_sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')
        raw_sub_sockets.append(raw_sub_socket)

    anno_socket = context.socket(zmq.PUB)
    anno_socket.bind(f"tcp://{yolo_config['ip']}:{yolo_config['anno_port']}")

    crop_socket = context.socket(zmq.PUB)
    crop_socket.bind(f"tcp://{yolo_config['ip']}:{yolo_config['crop_port']}")

    poller = zmq.Poller()
    for sock in raw_sub_sockets:
        poller.register(sock, zmq.POLLIN)

    while True:
        try:
            socks = dict(poller.poll())
            anno_msgs = {}
            crop_msgs = {}

            for raw_sub_socket in raw_sub_sockets:
                if raw_sub_socket in socks and socks[raw_sub_socket] == zmq.POLLIN:
                    msg = raw_sub_socket.recv_json()
                    stream_name = msg['stream_name']
                    frame = np.array(msg['frame'], dtype=np.uint8)
                    timestamp = msg['timestamp']

                    # Calculate scaled image size using scale factor
                    original_height, original_width = frame.shape[:2]
                    scale_factor = yolo_config['scale_factor']
                    scaled_image_size = (int(original_width / scale_factor), int(original_height / scale_factor))

                    results = model(
                        frame,
                        half=yolo_config['infer_half_precision'],
                        conf=yolo_config['confidence_threshold'],
                        iou=yolo_config['iou_threshold'],
                        imgsz=scaled_image_size,  # Use scaled image size
                        device=yolo_config['device'],
                        max_det=yolo_config['max_detections'],
                        vid_stride=yolo_config['frame_stride'],
                        stream_buffer=yolo_config['buffer_streams'],
                        visualize=yolo_config['visualize_features'],
                        augment=yolo_config['enable_augmentation'],
                        agnostic_nms=yolo_config['agnostic_nms_enabled'],
                        classes=yolo_config['target_classes'],
                        retina_masks=yolo_config['high_res_masks'],
                        embed=yolo_config['feature_layers'],
                        verbose=False
                    )

                    annotated_frame = results[0].plot()
                    cropped_frames = [
                        frame[int(box.xyxy[0, 1]):int(box.xyxy[0, 3]), int(box.xyxy[0, 0]):int(box.xyxy[0, 2])]
                        for result in results
                        for box in result.boxes
                    ]

                    anno_msg = {'frame': annotated_frame.tolist(), 'timestamp': timestamp}
                    crop_msg = {'frames': [cf.tolist() for cf in cropped_frames], 'timestamp': timestamp}

                    anno_msgs[stream_name] = anno_msg
                    crop_msgs[stream_name] = crop_msg

            if anno_msgs:
                anno_socket.send_json(anno_msgs)
            if crop_msgs:
                crop_socket.send_json(crop_msgs)

        except Exception as e:
            print(f"A YOLO error occurred: {e}")
            time.sleep(1)
