## files

 - `videoID`: video identifier
 - `videoList`: list au videoID to process (.lst)
 - `wavePath`: path to the wave file
 - `audioFile`: audio file in wave format (.wav)
 - `speakerSegmentationReferencePath`: path to manual speaker segmentation (.atseg)
 - `speechNonSpeechSegmentation`: segmentation of the audio into speech non speech (.mdtm)
 - `speechTurnSegmentation`: segmentation of the audio into speech turns (.mdtm)
 - `linearClustering`: segmentation after the linear clustering (.mdtm)
 - `linearClusteringPath`: path to the linear clustering
 - `BICMatrix`: distance matrix between speech turn (.mat) [st1 st2 BICDistance]
 - `BICMatrixPath`: path to BIC matrix 
 - `probaMatrix` probability that 2 speech turns correspond to the same person (.mat) [faceID1 faceID2 probability]
 - `diarization` face dirization (.mdtm)
 - `segment.uem`: segment to process
 - `modelSpeechNonSpeech`: model to segment audio signal into speech non speech
 - `modelBICToProba`: model to convert BIC distance similarity into probability

## add path_to_source_code to the PYTHONPATH

export PYTHONPATH=$PYTHONPATH:path_to_source_code

## Learn segmenter model for speech nonspeech segmentation

or used the existing model: "Model/modelSpeechNonSpeech"

python 1_learn_model_speech_nonspeech.py `wavePath` `videoList` `speakerSegmentationReferencePath` `segment.uem` `modelSpeechNonSpeech` 

## Speech nonspeech segmentation

`modelSpeechNonSpeech` can be found in the folder "Model"

python 2_speech_nonspeech_segmentation.py `audioFile` `modelSpeechNonSpeech` `speechNonSpeechSegmentation`

Options:
  --min_dur_speech=<mds>       minimum duration of a speech segment (>0) [default: 1.0]
  --min_dur_non_speech=<mdns>  minimum duration of a nonspeech segment (>0) [default: 0.8]

## Speech turn segmentation

python 3_speech_turn_segmentation.py videoID `audioFile` `speechNonSpeechSegmentation` `speechTurnSegmentation`

Options:
  --penalty_coef=<pc>   penalty coefficient for BIC (>0.0) [default: 1.2]
  --min_duration=<md>   minimum duration of a speech turn (>0.0) [default: 1.0]

## Linear clustering

python 4_linear_bic_clustering.py videoID `audioFile` `speechTurnSegmentation` `linearClustering`

Options:
  --penalty_coef=<pc>   penalty coefficient for BIC (>0.0) [default: 2.4]
  --gap=<g>             maximum gap between 2 speech turns that can be merged (>0.0) [default: 0.8]  

## Compute BIC matrix

python 5_compute_BIC_matrix.py videoID `audioFile` `linearClustering` `BICMatrix`

## learn normalisation model

or used the existing model: "Model/modelBICToProba"

python 6_learn_normalisation_model.py `videoList` `linearClusteringPath` `BICMatrixPath` `speakerSegmentationReferencePath` `modelBICToProba`

## compute score normalized between speech turns

`modelBICToProba` can be found in the folder "Model"

python 7_normalisation_matrix.py `videoID` `BICMatrix` `modelBICToProba` `probaMatrix`

## compute facetrack clustering

python 8_diarization.py `videoID` `linearClustering` `probaMatrix` `diarization`

Options:
  --threshold=<t>  stop criterion of the agglomerative clustering [default: 0.28]





