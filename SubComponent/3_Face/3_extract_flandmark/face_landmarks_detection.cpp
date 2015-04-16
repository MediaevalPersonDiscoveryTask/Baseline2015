
#include <fstream> 
#include <map> 
#include <highgui.h>

#include "flandmark_detector.h"

using namespace std;

int main( int argc, char** argv ) 
{
    if (argc < 3)  {
        fprintf(stderr, "Usage: face_flandmark_detection <video> <facetrack> <output> <flandmark_model>\n");
        exit(1);
    }
    
    char * video_path = argv[1];
    char * face_detection_path = argv[2];
    char * output = argv[3];
    char * flandmark_model = argv[4];
    
    FLANDMARK_Model * model = flandmark_init(flandmark_model);
    double * current_landmarks_tmp = (double*)malloc(2*model->data.options.M*sizeof(double));  
    int * box = (int*)malloc(4*sizeof(int));

    CvCapture * video = cvCaptureFromFile(video_path);
    IplImage*  frame = cvQueryFrame( video );
    
    int nb_frames = (int) cvGetCaptureProperty(video, CV_CAP_PROP_FRAME_COUNT);
    vector<int> l_frame(nb_frames);
    int count_frame = 0;
    int num_frame;

    std::map< int, std::list< std::vector<int> > > map_frame_box;

    ifstream fin;   
    fin.open(face_detection_path);          // open a file

    while (!fin.eof()){                     //read facetrack file
        char buf[5000];
        fin.getline(buf, 5000);

        int n = 0;                          // a for-loop index        
        const char* token[5000] = {};       // initialize to 0
        token[0] = strtok(buf, " ");        // first token         
        if (token[0]) {                     // zero if line is blank
            for (n = 1; n < 500; n++) {
                token[n] = strtok(0, " ");  // subsequent tokens
                if (!token[n]) break;       // no more tokens
            }
        }
        if (n>0){
            vector<int> fb(5);
            fb[0] = atoi(token[1]);
            fb[1] = atoi(token[2]);
            fb[2] = atoi(token[3]);
            fb[3] = fb[1] + atoi(token[4]);
            fb[4] = fb[2] + atoi(token[5]);   
            map_frame_box[atoi(token[0])].push_back(fb); 
        }
    } 

    ofstream fout(output);    
    while (frame == cvQueryFrame(video)) { 
        num_frame = (int) cvGetCaptureProperty(video, CV_CAP_PROP_POS_FRAMES);  							

        l_frame[count_frame] = num_frame;
        count_frame++;

        if ( map_frame_box.find(num_frame) != map_frame_box.end() ) {
            IplImage *frame_grayscale = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
            cvCvtColor(frame, frame_grayscale, CV_BGR2GRAY);

            while(!map_frame_box[num_frame].empty()){
                vector<int> fb(5);
                fb = map_frame_box[num_frame].front();

                box[0] = fb[1];
                box[1] = fb[2];
                box[2] = fb[3];
                box[3] = fb[4]; 

                cout << num_frame << endl ;

                fout << num_frame ;
                for (int j = 0; j < 5; j++) fout << " " << fb[j];
                if(flandmark_detect(frame_grayscale, box, model, current_landmarks_tmp)) for (int j = 0; j < 2*model->data.options.M; j++) fout << " " << -1.0;
                else for (int j = 0; j < 2*model->data.options.M; j++) fout << " " << current_landmarks_tmp[j];
                fout << endl;

                map_frame_box[num_frame].pop_front();
            }
            cvReleaseImage(&frame_grayscale);            
        }            
    }
    cvReleaseImage(&frame);
    flandmark_free(model);
    fout.close();
}
