
# files: 

<videoID>: video identifier
<videoFile>: video file in avi format
<faceDetection>: face detection (.face) [frameID xmin ymin width height number_of_neighbors]
<shotSegmentation>: list of shot to process with their segmentation (.shot)
<faceTracking>: face tracks position in each frame (.facetrack) [frameID xmin ymin width height]
<facetrackSegmentation>: temporal face tracks segmentation (.seg) [faceID StartTime endTime startFrame endFrame]
<idx>: Indexes of the video files (.MPG.idx), to convert frame number of the video.avi read with opencv to timestamp [frameID typeFrame positionInTheVideo timestamp]
<flandmark>: facial landmark position (.flandmark) [frameID xmin yminn xmax ymax (x y)*8]
<faceTrackDescriptor>: central HoG descriptor projected by LDML in 100 dimension (.desc) [faceID nb_desc distance_central d*100]
<l2Matrix>: distance matrix between facetracks (.mat) [faceID1 faceID2 nbHoGFaceID1  nbHoGFaceID2 distCenterFaceID1 distCenterFaceID2 l2Distance]
<refFacePosition> face postion in the reference (.position) [frameStart endFrame annotatedFrame personName role pointsPosition]
<probaMatrix> probability that to face correspond to the same person (.mat) [faceID1 faceID2 probability]
<diarization> face dirization (.mdtm)

# add path_to_source_code to the PYTHONPATH

export PYTHONPATH=$PYTHONPATH:path_to_source_code

# face detection

python 1_faceDetection.py <videoFile> <faceDetection> haarcascade_frontalface_default.xml --shot_segmentation=<shotSegmentation>

# face tracking

python 2_face_tracking.py <videoFile> <shotSegmentation> <faceDetection> <faceTracking> <facetrackSegmentation> --idx=<idx>

# extract facial landmark

cd 3_extract_flandmark/
mkdir build
cd build
cmake ..
make 

./face_landmarks_detection <videoFile> <faceTracking> <flandmark> flandmark_model.dat

# compute central HoG descriptor projected by LDML for each facetrack

python 4_face_HoG_descriptor.py <videoFile> <flandmark> features_model.txt ldml.txt <faceTrackDescriptor>

# compute l2 distance between facetracks

python 5_compute_hvh_matrix.py <faceTrackDescriptor> <l2Matrix>

# Learn normalisation model

python 6_learn_normalisation_model.py <video_list> <faceTracking> <refFacePosition> model_normalisation_l2_to_proba 

# normalize l2 distance into probability

python 7_normalisation_matrix.py <l2Matrix> model_normalisation_l2_to_proba <probaMatrix>

# compute facetrack clustering

python 8_diarization.py <videoID> <facetrackSegmentation> <probaMatrix> <diarization> --threshold=0.5
