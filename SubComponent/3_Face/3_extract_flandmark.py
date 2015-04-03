"""
Find face landmark after face tracking
"""
import glob, os

for f in sorted(glob.glob("/Users/poignant/Documents/mount/video_REPERE/ns382665.ovh.net/qcompere/phase*/*/*.avi")):
    video = f.split('/')[-1].split('.')[0]
    print video

    path_bin = "./face_landmarks_detection_macos"
    f_facetrack = "/Users/poignant/Documents/mount/work1_poignant/facetracks/"+video+".avi.facetrack"
    fout = "/Users/poignant/Documents/mount/work1_poignant/flandmark/"+video+".avi.flandmark"

    if not os.path.isfile(fout) and os.path.isfile(f_facetrack):
        os.popen(path_bin+" "+f+" "+f_facetrack+" "+fout)


