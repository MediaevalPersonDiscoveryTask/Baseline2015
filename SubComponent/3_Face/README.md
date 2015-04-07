
# face detection

python /path_to_source_code/SubComponent/3_face/1_face_detection.py path_to_video/video.avi path_to_metadata/3_Face/detection/video.face path_to_source_code/SubComponent/Model/haarcascade_frontalface_default.xml 

# face tracking

python /path_to_source_code/SubComponent/3_face/2_face_tracking.py path_to_video/video.avi path_to_metadata/1_Shot/segmentation/video.shot  path_to_metadata/3_Face/detection/video.face path_to_metadata/3_Face/tracking/video.facetrack path_to_metadata/3_Face/facetracks_segmentation/video.seg

# extract flandmark

cd /path_to_source_code/SubComponent/3_face/3_extract_flandmark/
mkdir build
cd build
cmake ..
make 

./path_to_source_code/SubComponent/3_face/3_extract_flandmark/build/face_landmarks_detection path_to_video/video.avi path_to_metadata/3_Face/tracking/video.facetrack path_to_metadata/3_Face/flandmark/video.flandmark path_to_source_code/SubComponent/Model/flandmark_model.dat

# Compute HoG descriptor projected by LDML for each facetrack

python /path_to_source_code/SubComponent/3_face/4_face_HoG_descriptor.py path_to_video/video.avi path_to_metadata/3_Face/flandmark/video.flandmark path_to_source_code/SubComponent/Model/features_model.txt path_to_source_code/SubComponent/Model/ldml.txt path_to_metadata/3_Face/facetraks_desc/video.desc

# Compute distance between face track

python /path_to_source_code/SubComponent/3_face/5_compute_hvh_matrix.py  path_to_metadata/3_Face/facetraks_desc/video.desc /path_to_source_code/SubComponent/3_face/Face_vs_Face_matrix/video.mat
