The HELMET dataset contains:
- “image" folder: contains 910 annotated video clips, each video clip has 100 frames (10 second x 10 fps) with 1920 x 1080 resolution.
- “data_split.csv”: contains indices of data splitting for training, validation and testing. 
- “annotation.zip”: contains annotation for each video clips, where the name of each annotation file (.csv) corresponds to the video clip in the “image” folder. In each annotation file, track_id corresponds to the unique tracking id of a motorcycle, frame_id correspond to the frame number (1 to 100) it appears, and with a bounding box (x, y, width, height) and annotated helmet use class. 
- “F-measure.zip”: contains the matlab scripts and an example about how to evaluate weighted F-measure of a proposed model.