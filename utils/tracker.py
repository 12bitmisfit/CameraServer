import os
import torch
import torchreid
from datetime import datetime
from PIL import Image
import time


def save_image_and_tensor(image, tensor, folder_path, file_prefix):
    os.makedirs(folder_path, exist_ok=True)
    img_path = os.path.join(folder_path, f'{file_prefix}.jpg')
    tensor_path = os.path.join(folder_path, f'{file_prefix}.pt')
    Image.fromarray(image).save(img_path)
    torch.save(tensor, tensor_path)


def load_people(base_path='./db/'):
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

    return people


def reid_process(tracker_config, shared_cropped_frames, shared_tracking_frames, lock):
    extractor = torchreid.utils.FeatureExtractor(
        tracker_config['model_name'],
        tracker_config['model_path'],
        tracker_config['device']
    )
    if tracker_config['dataset_dir'][:-1] == "/":
        people = load_people(tracker_config['dataset_dir'])
    else:
        people = load_people(tracker_config['dataset_dir'] + "/")
    known_people = people['known']
    unknown_people = people['unknown']

    while True:
        try:
            with lock:
                local_frames = shared_cropped_frames.copy()

            local_tracking = {stream_name: {'people': [], 'streams': []} for stream_name in local_frames.keys()}

            for stream_name, data in local_frames.items():
                cropped_frames = data
                if cropped_frames:
                    features = extractor(cropped_frames)
                    normalized_features = torch.nn.functional.normalize(features, p=2, dim=1)

                    for idx, (cropped_image, feature) in enumerate(zip(cropped_frames, normalized_features)):
                        now = datetime.now().strftime("%m_%d_%y_%H%M")
                        max_similarity = 0
                        best_match = None
                        folder_type = 'unknown'

                        for known_person, known_features in known_people.items():
                            similarity = torch.mm(feature.unsqueeze(0), known_features.t())
                            current_similarity = similarity.max().item()
                            if current_similarity > max_similarity:
                                max_similarity = current_similarity
                                best_match = known_person

                        if max_similarity > tracker_config['known_threshold']:
                            folder_type = 'known'
                        else:
                            for unknown_person, unknown_features in unknown_people.items():
                                similarity = torch.mm(feature.unsqueeze(0), unknown_features.t())
                                current_similarity = similarity.max().item()
                                if current_similarity > max_similarity:
                                    max_similarity = current_similarity
                                    best_match = unknown_person

                            if max_similarity < tracker_config['unknown_threshold']:
                                best_match = f'unknown{len(unknown_people) + 1}'
                                unknown_people[best_match] = feature.unsqueeze(0)
                                folder_type = 'unknown'

                        if max_similarity < tracker_config['no_update_threshold']:
                            folder_path = f'{tracker_config["dataset_dir"]}/{folder_type}/{best_match}'
                            save_image_and_tensor(cropped_image, feature, folder_path, now)

                            if folder_type == 'known':
                                if best_match in known_people:
                                    known_people[best_match] = torch.cat((known_people[best_match], feature.unsqueeze(0)), dim=0)
                                else:
                                    known_people[best_match] = feature.unsqueeze(0)
                            else:
                                if best_match in unknown_people:
                                    unknown_people[best_match] = torch.cat((unknown_people[best_match], feature.unsqueeze(0)), dim=0)
                                else:
                                    unknown_people[best_match] = feature.unsqueeze(0)

                        local_tracking[stream_name]['people'].append((best_match, feature, max_similarity))

            stream_names = list(local_tracking.keys())
            num_streams = len(stream_names)
            for i, stream_name in enumerate(stream_names):
                for offset in range(1, num_streams):
                    j = (i + offset) % num_streams
                    other_stream = stream_names[j]
                    for person, feature, similarity in local_tracking[stream_name]['people']:
                        for other_person, other_feature, other_similarity in local_tracking[other_stream]['people']:
                            if person == other_person:
                                cosine_similarity = torch.nn.functional.cosine_similarity(feature, other_feature, dim=0).item()
                                if cosine_similarity > tracker_config['stream_threshold']:
                                    local_tracking[stream_name]['streams'].append((other_stream, cosine_similarity))
                                    local_tracking[other_stream]['streams'].append((stream_name, cosine_similarity))

        except Exception as e:
            print(f"A tracking error occurred: {e}")
            time.sleep(1)
            continue

        with lock:
            shared_tracking_frames.update(local_tracking)
