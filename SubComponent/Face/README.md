
## files

 - `videoID`: video identifier
 - `videoList`: list au videoID to process (.mdtm)
 - `videoFile`: video file in avi format
 - `faceDetection`: face detection (.face) [frameID xmin ymin width height number_of_neighbors]
 - `shotSegmentation`: list of shot to process with their segmentation (.shot)
 - `rawFaceTrackPosition`: face tracks position in each frame (.facetrack) before the selection
 - `rawFaceTrackSegmentation`: temporal face tracks segmentation before the selection (.MESeg)
 - `faceTrackPosition`: face tracks position in each frame after the selection (.facetrack)
 - `faceTrackSegmentation`: temporal face tracks segmentation (.MESeg) after the selection
 - `idx`: Indexes of the video files (.idx), to convert frame number of the video read with opencv to timestamp
 - `flandmark`: facial landmark position (.flandmark)
 - `faceTrackDescriptor`: central HoG descriptor projected by LDML in 100 dimension (.desc)
 - `l2Matrix`: distance matrix between facetracks (.mat)
 - `l2MatrixPath`: path to l2 distance matrix
 - `facePositionReference`: path to face position in the reference (.position) [frameStart endFrame annotatedFrame personName role pointsPosition]
 - `probaMatrix`: probability that 2 facetracks correspond to the same person (.mat)
 - `diarization`: face diarization (.MESeg)

file format can be found [here](https://github.com/MediaevalPersonDiscoveryTask/metadata/wiki/file-format)

 #### Model:

 - `haarcascade_frontalface_default.xml`: haar cascasde used by opencv to find face
 - `flandmark_model.dat`: facial landmark model
 - `features_model.txt`: features model for affine transformation
 - `ldml.txt`: projection matrix of the HoG descriptor into a smaller dimension (100)
 - `modell2ToProba`: model to convert l2 distance to probability

## add path_to_source_code to the PYTHONPATH

export PYTHONPATH=$PYTHONPATH:path_to_source_code

## face detection

python 1_faceDetection.py `videoID` `videoFile` `faceDetection_out` `haarcascade_frontalface_default.xml` --shot_segmentation=`shotSegmentation`

## face tracking

If the timestamps in the `faceTrackSegmentation` output are not good (like 0.0 0.0), you can try to used an idx file generate by the script "mediaeval_util/extract_idx.py"

python 2_face_tracking.py `videoID` `videoFile` `shotSegmentation` `FaceDetection` `rawFaceTrackPosition_out` `rawFaceTrackSegmentation_out` --idx=`idx`

## extract a descriptor per face track to filter out bad face tracks

Change the video width with the good value

python 3_extract_desc_speaking_face.py `videoID` `rawFaceTrackPosition` `rawFaceTrackSegmentation` `descFaceSelection_out` --videoWidth=1024

## learn a model to filter out bad face tracks

or used the existing model: "Model/model_face_selection"

python 4_learn_model_proba_speaking_face.py `videoList` `rawFaceTrackPositionPath` `descFaceSelection` `facePositionReference` `modelFaceSelection_out`

## select face tracks

`modelFaceSelection` can be found in the folder "Model"

python 5_selection_facetrack.py `descFaceSelection` `rawFaceTrackPosition` `rawFaceTrackSegmentation` `faceTrackPosition_out` `faceTrackSegmentation_out` `modelFaceSelection` --thr=`t` --minDuration=`md`

## extract facial landmarks

First compile the binary:

cd 6_extract_flandmark/
mkdir build
cd build
cmake ..
make 

`flandmark_model.dat` can be found in the folder "Model"


./face_landmarks_detection `videoFile` `faceTrackPosition` `flandmark_out` `flandmark_model.dat`

## compute central HoG descriptor projected by LDML for each facetrack

`features_model.txt` `ldml.txt` can be found in the folder "Model"

python 7_facetrack_descriptor.py `videoFile` `flandmark` `features_model.txt` `ldml.txt` `faceTrackDescriptor_out`

## compute l2 distance between facetracks

python 8_l2_matrix.py `faceTrackDescriptor` `l2Matrix_out` `faceTrackSegmentation`

## Learn normalisation model

or used the existing model: "Model/modell2ToProba"

python 9_learn_normalisation_model.py `video_list` `faceTrackPositionPath`  `l2MatrixPath` `facePositionReference` `modell2ToProba_out` 

## normalize l2 distance into probability

`modell2ToProba` can be found in the folder "Model"

python 10_normalisation_matrix.py `l2Matrix` `modell2ToProba` `probaMatrix_out`

## compute facetrack clustering

python 11_diarization.py `videoID` `facetrackSegmentation` `probaMatrix` `diarization_out`
