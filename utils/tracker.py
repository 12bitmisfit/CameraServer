import torch
import torchreid

# A dictionary representing a structure for tracking information in multiple video streams
'''
local_tracking = {
    "stream1": {  # Multiple streams can be tracked, each identified uniquely
        'people': {  # Key to store information about detected people
            'person1':  {  # Each person is indexed uniquely within the stream
                'cropped_image':  # Stores the cropped image of the person
                'embedding':  # Stores the feature embedding vector of the person
                'cosine_similarity_map': {
                    'stream2': [] # Array storing cosine similarities with people in other streams
                }
            }
        }
    }
}
'''


def reid_process(tracker_config, shared_cropped_frames, shared_tracking_frames, lock):

    # Initialize the feature extractor
    extractor = torchreid.utils.FeatureExtractor(
        model_name='osnet_ain_x1_0',
        model_path='./models/reid/osnet_ain_x1_0.pth',
        device='cuda:0'
    )

    # Local dictionary to store tracking info for each stream
    local_tracking = {}

    # Make a local copy of shared_cropped_frames for thread safety
    with lock:
        local_frames = shared_cropped_frames.copy()

    stream_names = list(local_frames.keys())

    # Initialize the local_tracking dictionary structure for each stream
    for stream_name in stream_names:
        local_tracking[stream_name] = {'people': {}}
        for idx, cropped_image in enumerate(local_frames[stream_name]['cropped_frames']):
            person_key = f'person{idx + 1}'
            local_tracking[stream_name]['people'][person_key] = {
                'cropped_image': cropped_image,
                'embedding': None,
                'cosine_similarity_map': {}
            }

    # Main loop to continuously process frames
    while True:
        # Extract features for all cropped images in each stream
        for stream_name in local_frames.keys():
            if local_frames[stream_name]['cropped_frames']:
                features = extractor(local_frames[stream_name]['cropped_frames'])
                features_cpu = features.cpu()
                features_numpy = features_cpu.numpy()

                # Update person embeddings with extracted features for each detected individual in the stream
                for idx, (feature, person_key) in enumerate(
                        zip(features_numpy, local_tracking[stream_name]['people'].keys())):
                    local_tracking[stream_name]['people'][person_key]['embedding'] = feature

        # Compute cosine similarities between features of different streams
        for i in range(len(stream_names)):
            for j in range(i + 1, len(stream_names)):
                stream_i = stream_names[i]
                stream_j = stream_names[j]
                for person_i_key, person_i in local_tracking[stream_i]['people'].items():
                    for person_j_key, person_j in local_tracking[stream_j]['people'].items():
                        if person_i['embedding'] is not None and person_j['embedding'] is not None:
                            feature_i = torch.tensor(person_i['embedding'], device='cuda:0').unsqueeze(0)
                            feature_j = torch.tensor(person_j['embedding'], device='cuda:0').unsqueeze(0)
                            feature_i_norm = torch.nn.functional.normalize(feature_i, p=2, dim=1)
                            feature_j_norm = torch.nn.functional.normalize(feature_j, p=2, dim=1)
                            cosine_sim = torch.mm(feature_i_norm, feature_j_norm.T)
                            cosine_similarity = cosine_sim.cpu().numpy().flatten().tolist()
                            if stream_j not in person_i['cosine_similarity_map']:
                                person_i['cosine_similarity_map'][stream_j] = []
                            person_i['cosine_similarity_map'][stream_j].extend(cosine_similarity)

        # Update the shared data structure with newly calculated tracking information
        # todo: possibly update so it resets local_tracking each time
        with lock:
            shared_tracking_frames = local_tracking
            local_frames = shared_cropped_frames.copy()
