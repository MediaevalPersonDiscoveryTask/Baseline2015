## files

 - `videoFile`: video file in avi format
 - `flandmark`: facial landmark position (.flandmark) [frameID xmin yminn xmax ymax (x y)*8]
 - `SpeakingFaceDescriptor`: visual descriptor of speaking face
 - `SpeakingFaceDescriptorPath`: path to visual descriptor of speaking face
 - `videoList`: list au videoID to process (.lst)
 - `idx`: indexes of the video files (.MPG.idx), to convert frame number of the video.avi read with opencv to timestamp [frameID typeFrame positionInTheVideo timestamp]
 - `idxPath`: path to the idx file
 - `faceTrackPositionPath`: face tracks position in each frame (.facetrack) [frameID xmin ymin width height]
 - `facePositionReferencePath`: face postion in the reference (.position) [frameStart endFrame annotatedFrame personName role pointsPosition]
 - `speakerSegmentationReferencePath`: path to manual speaker segmentation (.atseg)
 - `speechTurnSegmentation`: speech turn segmentation (.mdtm)
 - `probaSpeakingFace` probability that a face track speaks (.mat) [st faceID probability]
 - `modelProbaSpeakingFace`: classifier model to compute the probabilty that a facetrack correspond to the current speech turn

## add path_to_source_code to the PYTHONPATH

export PYTHONPATH=$PYTHONPATH:path_to_source_code

## extract visual speaking face descriptor

python 1_extract_desc_speaking_face.py `videoFile` `flandmark` `SpeakingFaceDescriptor`

## learn a model to find speaking face

or used the existing model: "Model/modelProbaSpeakingFace"

python 2_learn_model_proba_speaking_face.py `videoList` `idxPath` `faceTrackPositionPath` `SpeakingFaceDescriptorPath` `facePositionReferencePath` `speakerSegmentationReferencePath` `modelProbaSpeakingFace`

## Compute score between face tracks and speech turns

`modelProbaSpeakingFace` can be found in the folder "Model"

python 3_proba_speaking_face.py `videoID` `faceTrackSegmentation` `SpeakingFaceDescriptor` `speechTurnSegmentation` `idx` `modelProbaSpeakingFace` `probaSpeakingFace`

## complete the speaking face matrix for no co-occurring speech turns/face tracks

python 4_complete_speakingFace_matrix.py `videoID` `speechTurnSegmentation` `faceTrackSegmentation` `probaSpeechTurns` `probaFacetracks` `probaSpeakingFace` `output_matrix`


