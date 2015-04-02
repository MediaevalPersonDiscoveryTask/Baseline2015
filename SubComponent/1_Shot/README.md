
1. Extract a descriptor to find shot boundaries: 

& python 1_extract_descriptor.py path_to_video/video.avi data_path/1_shot/descriptor/segmentation/video.desc --x=25 --y=25 --w=975 --h=400


2. Find shot boundaries

& python 2_segmentation descriptor data_path/1_shot/descriptor/video.desc data_path/1_shot/segmentation/video.cut

