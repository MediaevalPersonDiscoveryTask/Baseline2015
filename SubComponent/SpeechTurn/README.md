### files

 - `videoID`: video identifier
 - `videoList`: list au videoID to process (.lst)
 - `wavePath`: path to the wave file
 - `audioFile`: audio file in wave format (.wav)
 - `refspeakerSegmentationPath`: path to manual speaker segmentation (.atseg)
 - `speechNonSpeechSegmentation`: segmentation of the audio into speech non speech (.mdtm)
 - `speechTurnSegmentation`: segmentation of the audio into speech turns (.mdtm)
 - `linearClustering`: segmentation after the linear clustering (.mdtm)
 - `linearClusteringPath`: path to the linear clustering
 - `BICMatrix`: distance matrix between speech turn (.mat) [st1 st2 BICDistance]
 - `BICMatrixPath`: path to BIC matrix 
 - `probaMatrix` probability that 2 speech turns correspond to the same person (.mat) [faceID1 faceID2 probability]
 - `diarization` face dirization (.mdtm)

## add path_to_source_code to the PYTHONPATH

export PYTHONPATH=$PYTHONPATH:path_to_source_code

## Learn segmenter model for speech nonspeech segmentation

python 1_learn_model_speech_nonspeech.py `wavePath` `videoList` `refspeakerSegmentationPath` segment.uem modelSpeechNonSpeech 

## Speech nonspeech segmentation

python 2_speech_nonspeech_segmentation.py `audioFile` modelSpeechNonSpeech `speechNonSpeechSegmentation`

## Speech turn segmentation

python 3_speech_turn_segmentation.py videoID `audioFile` `speechNonSpeechSegmentation` `speechTurnSegmentation`

## Linear clustering

python 4_linear_bic_clustering.py videoID `audioFile` `speechTurnSegmentation` `linearClustering`

## Compute BIC matrix

python 5_compute_BIC_matrix.py videoID `audioFile` `linearClustering` `BICMatrix`

## learn normalisation model

python 6_learn_normalisation_model.py `videoList` `linearClusteringPath` `BICMatrixPath` `refspeakerSegmentationPath` normalisationModel

## compute score normalized between speech turns

python 7_normalisation_matrix.py `videoID` `linearClustering` `BICMatrix` normalisationModel `probaMatrix`

## compute facetrack clustering

python 8_diarization.py `videoID` `linearClustering` `probaMatrix` `diarization`






