
# Learn segmenter model for speech nonspeech segmentation

python 1_learn_model_speech_nonspeech.py path_to_wave/video.wav path_to_video_list/uri.dev.lst path_to_reference/speaker.dev.mdtm data_path/uem/segment.uem data_path/model/model_speech_nonspeech 

# Speech nonspeech segmentation

python 2_speech_nonspeech_segmentation.py video path_to_wave data_path/model/model_speech_nonspeech data_path/2_SpeechTurn/seg_speech_nonspeech --min_dur_speech=1.0 --min_dur_non_speech=0.8

# Speech turn segmentation

python 3_speech_turn_segmentation.py video path_to_wave data_path/2_SpeechTurn/seg_speech_nonspeech data_path/2_SpeechTurn/seg_speech_turn --penalty_coef=1.6 --minduration=0.8

# Linear clustering

python 4_linear_bic_clustering.py video path_to_wave data_path/2_SpeechTurn/seg_speech_turn data_path/2_SpeechTurn/seg_speech_turn_linear_clus --penalty_coef=1.8 --gap=0.8

# Compute BIC matrix

python 5_compute_BIC_matrix.py video path_to_wave data_path/2_SpeechTurn/seg_speech_turn_linear_clus data_path/2_SpeechTurn/matrix_BIC

# learn normalisation model

python 6_learn_normalisation_model.py path_to_video_list/uri.dev.lst data_path/2_SpeechTurn/seg_speech_turn_linear_clus data_path/2_SpeechTurn/matrix_BIC path_to_reference/speaker.dev.mdtm data_path/model/normalisation_model --min_cooc=0.5

# compute score normalized between speech turn

python 7_normalisation_matrix.py video data_path/2_SpeechTurn/seg_speech_turn_linear_clus data_path/2_SpeechTurn/matrix_BIC data_path/model/normalisation_model data_path/2_SpeechTurn/matrix_svs







