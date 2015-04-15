
# add path_to_source_code to the PYTHONPATH

export PYTHONPATH=$PYTHONPATH:path_to_source_code

# Learn segmenter model for speech nonspeech segmentation

python 1_learn_model_speech_nonspeech.py wave_path video_list Spk_seg_ref_path segment.uem model_speech_nonspeech 

# Speech nonspeech segmentation

python 2_speech_nonspeech_segmentation.py videoID.wav model_speech_nonspeech seg_speech_nonspeech/videoID.mdtm --min_dur_speech=1.0 --min_dur_non_speech=0.8

# Speech turn segmentation

python 3_speech_turn_segmentation.py videoID videoID.wav seg_speech_nonspeech/videoID.mdtm speech_turn_segmentation/videoID.mdtm --penalty_coef=1.6 --minduration=0.8

# Linear clustering

python 4_linear_bic_clustering.py videoID videoID.wav speech_turn_segmentation/videoID.mdtm linear_clustering/videoID.mdtm --penalty_coef=1.8 --gap=0.8

# Compute BIC matrix

python 5_compute_BIC_matrix.py videoID videoID.wav linear_clustering/videoID.mdtm matrix_BIC/videoID.mat

# learn normalisation model

python 6_learn_normalisation_model.py video_list linear_clustering_path matrix_BIC_path Spk_seg_ref_path spk_vs_spk_normalisation_model --min_cooc=0.5

# compute score normalized between speech turn

python 7_normalisation_matrix.py videoID linear_clustering/videoID.mdtm matrix_BIC/videoID.mat spk_vs_spk_normalisation_model matrix_normalized/videoID.mat







