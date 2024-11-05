# CameraServer
personal security camera system with people detection and re-identification


## Current working (or mostly working) Features:
- Accepts multiple video streams from any source supported by cv2 (only IP and usb cameras tested)
- Starts a process per video source that handles reading and writing video frames
- configurable output fps when writing to disk
- configurable output resolution when writing to disk
- configurable processing frame rate for YOLO detection and tracking to reduce resource usage
- configurable length (in seconds) of video files written to disk
- configurable YOLO model (ultralytics is the back end, get models from them)
- configurable scaling of input image size to the YOLO model to reduce resource usage
- configurable device placement for ReID and YOLO models (cuda:0, cpu, etc)
- configurable thresholds for matching known and unknown people
- configurable threshold for matching people visible in multiple video streams
- dynamic person identification and re-identification (ReID) via torchreid
- realtime generation of a dataset containing feature embeddings and cropped images
-  Supports known and unknown people, including adding feature embeddings and cropped images
-  configurable thresholds for matching people in frame to known or unknown people
-  generates new id's for new unknown people
-  utilize a cosine similarity map to identify people visible in multiple video streams


## What needs fixed or implemented:
- change dataset structure to something better than cropped images and saved tensors
- add pan + tilt support for cameras
- add web interface for viewing the system
- add configuration modification through web interface?
- add hooks to tie into home assistant (provide states of who is in what room, etc)
- add support for non pose varients of ultralytic's YOLO models
- add support for tracking objects that aren't humans (cars come to mind)
- add support for running multiple feature extration models

## Whats been updated:

### 11/4/24
- Rolled back to older version with cosine similarity map and shared memory instead of ZMQ
- Added a re-broadcast function that re-broadcasts the input video streams to reduce network traffic
- Removed the half baked flask stuff
- Added a basic web interface for viewing all the streams. It is served via fastapi.

### Old
- removed cosine similarity map
- rewrote code to use pyzmq instead of shared memory | **Preparing for front end work**
- update tracker.py and main.py to use tracker.yaml for configurable items | **added a reid.yaml file with a few options**
- clean up a lot of the dirty/dumb data handling that was implemented to make things functional | **kinda just made things more readable**
- make dataset creation + updating more configurable and more scalable | **added a no update threshold to mitigate excess bloat + small fixes**
- modify cosine similarity calculations | **fixed a lot, including adding a cosine similarity map + threshold for those viewable in multiple streams**

## What is of concern and will most likely need addressed:
- pytorch tensors can be stored on the GPU, may need to explicitly move them to CPU when adding additional functionality in the future
- known + unknown people will have lots and lots of feature embeddings added over time, will almost certainly need to add more configurable limits and optimize things (perhaps pruning based on age of embeddings and/or frequency of person seen?)
- planned pan + tilt support is specific to my use case and is not universal/standardized (An esp32 + 2x 28BYJ-48 stepper motors with ULN2003 drivers and a custom android application to forward commands received over wifi to the esp32 via bluetooth)

## Long term goals:
- add support for training/fine-tuning feature extractors
- add support for training/fine-tuning object detectors
- make a more portable version for easy deployment (likely docker)
