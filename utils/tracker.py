import os
import torch
import torchreid
from datetime import datetime
from PIL import Image
import time
import zmq
import numpy as np


def save_image_and_tensor(image, tensor, folder_path, file_prefix):
    print("saved something")
    os.makedirs(folder_path, exist_ok=True)
    img_path = os.path.join(folder_path, f'{file_prefix}.jpg')
    tensor_path = os.path.join(folder_path, f'{file_prefix}.pt')
    Image.fromarray(image).save(img_path)
    torch.save(tensor, tensor_path)


def load_people(base_path='./db/'):
    print("loading people")
    people = {'known': {}, 'unknown': {}}
    known_path = os.path.join(base_path, 'known')
    unknown_path = os.path.join(base_path, 'unknown')

    os.makedirs(known_path, exist_ok=True)
    os.makedirs(unknown_path, exist_ok=True)

    for folder_name in os.listdir(known_path):
        person_path = os.path.join(known_path, folder_name)
        if os.path.isdir(person_path):
            tensors = []
            for file in os.listdir(person_path):
                if file.endswith('.pt'):
                    tensor_path = os.path.join(person_path, file)
                    tensors.append(torch.load(tensor_path))
            if tensors:
                people['known'][folder_name] = torch.stack(tensors)

    for folder_name in os.listdir(unknown_path):
        person_path = os.path.join(unknown_path, folder_name)
        if os.path.isdir(person_path):
            tensors = []
            for file in os.listdir(person_path):
                if file.endswith('.pt'):
                    tensor_path = os.path.join(person_path, file)
                    tensors.append(torch.load(tensor_path))
            if tensors:
                people['unknown'][folder_name] = torch.stack(tensors)

    print("loaded people")
    return people


def reid(reid_config):
    extractor = torchreid.utils.FeatureExtractor(
        model_name=reid_config['model_name'],
        model_path=reid_config['model_path'],
        device=reid_config['device']
    )
    if reid_config['dataset_dir'][:-1] == "/":
        people = load_people(reid_config['dataset_dir'])
    else:
        people = load_people(reid_config['dataset_dir'] + "/")

    context = zmq.Context()

    crop_sub_socket = context.socket(zmq.SUB)
    crop_sub_socket.connect(f"tcp://{reid_config['ip']}:{reid_config['crop_port']}")
    crop_sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')

    track_socket = context.socket(zmq.PUB)
    track_socket.bind(f"tcp://{reid_config['ip']}:{reid_config['track_port']}")

    print("sockets done")

    while True:
        try:
            crop_msgs = crop_sub_socket.recv_json()
            live_tracks = dict.fromkeys(crop_msgs, [])
            live_data = None
            for stream_name in crop_msgs.keys():
                frames = [np.array(f, dtype=np.uint8) for f in crop_msgs[stream_name]['frames']]
                timestamp = crop_msgs[stream_name]['timestamp']
                if frames:
                    live_data = []
                    features = extractor(frames)
                    normalized_features = torch.nn.functional.normalize(features, p=2, dim=1)

                    for cropped_image, norm_feature in zip(frames, normalized_features):
                        now = datetime.now().strftime("%m_%d_%y_%H%M%S_%f")[:-3]
                        max_similarity = 0
                        best_match = None
                        folder_type = 'unknown'

                        for known_person, known_features in people['known'].items():
                            similarity = torch.mm(norm_feature.unsqueeze(0), known_features.t())
                            current_similarity = similarity.max().item()
                            if current_similarity > max_similarity:
                                max_similarity = current_similarity
                                best_match = known_person

                        if max_similarity > reid_config['known_threshold']:
                            folder_type = 'known'
                        else:
                            for unknown_person, unknown_features in people['unknown'].items():
                                similarity = torch.mm(norm_feature.unsqueeze(0), unknown_features.t())
                                current_similarity = similarity.max().item()
                                if current_similarity > max_similarity:
                                    max_similarity = current_similarity
                                    best_match = unknown_person

                            if max_similarity < reid_config['unknown_threshold']:
                                best_match = f'unknown{len(people["unknown"]) + 1}'
                                people['unknown'][best_match] = norm_feature.unsqueeze(0)
                                folder_type = 'unknown'

                        live_data.append((stream_name, best_match, norm_feature))
                        print(f"best match: {best_match} | sim: {max_similarity}")

                        if max_similarity < reid_config['no_update_threshold']:
                            folder_path = f'{reid_config["dataset_dir"]}/{folder_type}/{best_match}'
                            save_image_and_tensor(cropped_image, norm_feature, folder_path, now)

                            if folder_type == 'known':
                                if best_match in people['known']:
                                    people['known'][best_match] = torch.cat((people['known'][best_match], norm_feature.unsqueeze(0)), dim=0)
                                else:
                                    people['known'][best_match] = norm_feature.unsqueeze(0)
                            else:
                                if best_match in people['unknown']:
                                    people['unknown'][best_match] = torch.cat((people['unknown'][best_match], norm_feature.unsqueeze(0)), dim=0)
                                else:
                                    people['unknown'][best_match] = norm_feature.unsqueeze(0)

            # Removed cosine similarity map due to computational requirements
            if live_data:
                for stream_name, best_match, norm_feature in live_data:
                    if best_match not in live_tracks[stream_name]:
                        live_tracks[stream_name].append(best_match)
                track_socket.send_json(live_tracks)

        except Exception as e:
            print(f"A tracking error occurred: {e}")
            time.sleep(1)
