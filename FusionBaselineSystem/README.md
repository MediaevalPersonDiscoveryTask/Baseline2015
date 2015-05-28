
# late_naming

 The late naming process try to propagate written names onto speaker clusters and then find if there is a co-occurring face with a probability to be the same person as the speaker higher than a threshold.

 It can be divided in 6 steps : 

 1. Select written names that can be used as evidence (only those that co-occurring a shot in the list of shot submissions).
 2. Propagate written names onto speaker cluster with first an mapping 1-to-1 based on co-occurrence duration (HungarianTagger function) and then name speakers that co-occurring a written names (direct function).
 3. Find for each speaker if there is a co-occurring face that has a probability to correspond to the current speaker higher than a threshold.
 4. For each faces found, name them with the name of the current speaker.
 5. For each shot in the submission list, find if there is a speaker and a face with the same name associated that speak/appear in the shot. If this is the case, check also if there is an evidence that correspond to this name.
 6. Compute the confidence for each hypothesis : 
    - Find the written name closest temporally
    - If the hypothesis co-occur the written name, confidence = 1.0 + co-occurrence duration
    - Else, confidence = 1.0/(minimum temporal gap between the hypothesis and the written name)

    