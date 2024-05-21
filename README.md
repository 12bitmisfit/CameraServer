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
- dynamic person identification and re-identification (ReID) via torchreid
- realtime generation of a dataset containing feature embeddings and cropped images
-  Supports known and unknown people, including adding feature embeddings and cropped images
-  configurable thresholds for matching people in frame to known or unknown people
-  generates new id's for new unknown people


## What needs fixed or implemented:
- update tracker.py and main.py to use tracker.yaml for configurable items
- clean up a lot of the dirty/dumb data handling that was implemented to make things functional
- make dataset creation + updating more configurable and more scalable
- modify cosine similarity calculations
- add pan + tilt support for cameras
- add web interface for viewing the system
- add configuration modification through web interface?
- add hooks to tie into home assistant (provide states of who is in what room, etc)
- add support for non pose varients of ultralytic's YOLO models
- add support for tracking objects that aren't humans (cars come to mind)
- add support for running multiple feature extration models


## Long term goals:
- add support for training/fine-tuning feature extractors
- add support for training/fine-tuning object detectors
- make a more portable version for easy deployment (likely docker)
