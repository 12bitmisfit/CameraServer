from ultralytics import YOLO
import time


def yolo(yolo_config, shared_raw_frames, shared_annotated_frames, shared_cropped_frames, lock):
    model = YOLO(yolo_config['model_path'], verbose=False)
    time_tracking = {}
    cropped_dict = {}
    annotated_dict = {}

    while True:
        try:
            with lock:
                local_frames = shared_raw_frames.copy()

            for stream_name in local_frames.keys():
                time_tracking[stream_name] = local_frames[stream_name]['raw_frame'][1]
                cropped_dict[stream_name] = {"cropped_images": []}
                annotated_dict[stream_name] = {"annotated_images": []}
            break

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1)

    while True:
        try:
            for stream_name in local_frames.keys():
                if local_frames[stream_name]['raw_frame'][1] > time_tracking[stream_name]:
                    original_height, original_width = local_frames[stream_name]['raw_frame'][0].shape[:2]
                    scale_factor = yolo_config['scale_factor']
                    scaled_image_size = (int(original_width / scale_factor), int(original_height / scale_factor))

                    results = model(
                        local_frames[stream_name]['raw_frame'][0],
                        half=yolo_config['infer_half_precision'],
                        conf=yolo_config['confidence_threshold'],
                        iou=yolo_config['iou_threshold'],
                        imgsz=scaled_image_size,
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

                    annotated_dict[stream_name]['annotated_frame'] = (
                        results[0].plot(), local_frames[stream_name]['raw_frame'][0]
                    )

                    cropped_dict[stream_name]['cropped_frames'] = [
                        local_frames[stream_name]['raw_frame'][0][int(box.xyxy[0, 1]):int(box.xyxy[0, 3]), int(box.xyxy[0, 0]):int(box.xyxy[0, 2])]
                        for result in results
                        for box in result.boxes
                    ]

            with lock:
                for stream_name in local_frames.keys():
                    shared_annotated_frames[stream_name] = annotated_dict[stream_name]['annotated_frame']
                    shared_cropped_frames[stream_name] = cropped_dict[stream_name]['cropped_frames']
                local_frames = shared_raw_frames.copy()

        except Exception as e:
            print(f"A YOLO error occurred: {e}")
            time.sleep(1)
