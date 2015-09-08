
## files

 - `videoID`: video identifier
 - `videoList`: list au videoID to process (.mdtm)
 - `videoFile`: video file in avi format
 - `faceDetection`: face detection (.face) [frameID xmin ymin width height number_of_neighbors]
 - `shotSegmentation`: list of shot to process with their segmentation (.shot)
 - `faceTrackPosition`: face tracks position in each frame (.facetrack) [frameID xmin ymin width height]
 - `faceTrackSegmentation`: temporal face tracks segmentation (.seg) [faceID StartTime endTime startFrame endFrame]
 - `idx`: Indexes of the video files (.MPG.idx), to convert frame number of the video.avi read with opencv to timestamp [frameID typeFrame positionInTheVideo timestamp]
 - `flandmark`: facial landmark position (.flandmark) [frameID xmin yminn xmax ymax (x y)*8]
 - `faceTrackDescriptor`: central HoG descriptor projected by LDML in 100 dimension (.desc) [faceID nb_desc distance_central d*100]
 - `l2Matrix`: distance matrix between facetracks (.mat) [faceID1 faceID2 nbHoGFaceID1  nbHoGFaceID2 distCenterFaceID1 distCenterFaceID2 l2Distance]
 - `l2MatrixPath`: path to l2 distance matrix
 - `facePositionReferencePath`: path to face postion in the reference (.position) [frameStart endFrame annotatedFrame personName role pointsPosition]
 - `probaMatrix`: probability that 2 facetracks correspond to the same person (.mat) [faceID1 faceID2 probability]
 - `diarization`: face diarization (.mdtm)
 - `haarcascade_frontalface_default.xml`: haar cascasde used by opencv to find face
 - `flandmark_model.dat`: facial landmark model
 - `features_model.txt`: features model for affine transformation
 - `ldml.txt`: projection matrix of the HoG descriptor into a smaller dimension (100)
 - `modell2ToProba`: model to convert l2 distance to probability

## add path_to_source_code to the PYTHONPATH

export PYTHONPATH=$PYTHONPATH:path_to_source_code

## face detection

python 1_faceDetection.py `videoFile` `faceDetection` `haarcascade_frontalface_default.xml` --shot_segmentation=`shotSegmentation`

## face tracking

python 2_face_tracking.py `videoFile` `shotSegmentation` `faceDetection` `faceTrackPosition` `facetrackSegmentation` --idx=`idx`

## extract facial landmarks

cd 3_extract_flandmark/
mkdir build
cd build
cmake ..
make 

./face_landmarks_detection `videoFile` `faceTrackPosition` `flandmark` `flandmark_model.dat`

## compute central HoG descriptor projected by LDML for each facetrack

python 4_face_HoG_descriptor.py `videoFile` `flandmark` `features_model.txt` `ldml.txt` `faceTrackDescriptor`

## compute l2 distance between facetracks

python 5_compute_hvh_matrix.py `faceTrackDescriptor` `l2Matrix`

## Learn normalisation model

python 6_learn_normalisation_model.py `video_list` `faceTrackPosition`  `l2MatrixPath` `facePositionReferencePath` `modell2ToProba` 

## normalize l2 distance into probability

python 7_normalisation_matrix.py `l2Matrix` `modell2ToProba` `probaMatrix`

## compute facetrack clustering

python 8_diarization.py `videoID` `facetrackSegmentation` `probaMatrix` `diarization`
