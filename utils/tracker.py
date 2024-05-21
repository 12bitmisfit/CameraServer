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

    # Load known people
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

    # Load unknown people
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

def reid_process(tracker_config, shared_cropped_frames, shared_tracking_frames, lock, known_similarity_threshold=0.7, unknown_similarity_threshold=0.3):
    extractor = torchreid.utils.FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='./models/reid/osnet_x1_0.pth',
        device='cuda:0'
    )

    people = load_people()
    known_people = people['known']
    unknown_people = people['unknown']
    local_tracking = {}

    while True:
        try:
            with lock:
                local_frames = shared_cropped_frames.copy()

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

                        # Check against all known people
                        for known_person, known_features in known_people.items():
                            similarity = torch.mm(feature.unsqueeze(0), known_features.t())
                            current_similarity = similarity.max().item()
                            if current_similarity > max_similarity:
                                max_similarity = current_similarity
                                best_match = known_person

                        # Decide based on the highest similarity found
                        if max_similarity > known_similarity_threshold:
                            folder_type = 'known'
                        else:
                            # Check against unknown people
                            for unknown_person, unknown_features in unknown_people.items():
                                similarity = torch.mm(feature.unsqueeze(0), unknown_features.t())
                                current_similarity = similarity.max().item()
                                if current_similarity > max_similarity:
                                    max_similarity = current_similarity
                                    best_match = unknown_person

                            if max_similarity < unknown_similarity_threshold:
                                # Assign a new unknown identity
                                best_match = f'unknown{len(unknown_people) + 1}'
                                unknown_people[best_match] = feature.unsqueeze(0)
                                folder_type = 'unknown'

                        # Save image and tensor, update local tracking
                        folder_path = f'./db/{folder_type}/{best_match}'
                        save_image_and_tensor(cropped_image, feature, folder_path, now)

                        if best_match not in local_tracking:
                            local_tracking[best_match] = {
                                'cropped_image': cropped_image,
                                'streams': [stream_name]
                            }
                        else:
                            local_tracking[best_match]['streams'].append(stream_name)
        except Exception as e:
            print(f"A tracking error occurred: {e}")
            time.sleep(1)  # Optional: wait before retrying
            continue

        with lock:
            shared_tracking_frames.update(local_tracking)
