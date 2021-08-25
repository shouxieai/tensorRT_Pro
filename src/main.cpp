
#include <stdio.h>
#include <string.h>

int app_yolo();
int app_alphapose();
int app_fall_recognize();
int app_retinaface();
int app_arcface();
int app_arcface_video();
int app_arcface_tracker();

int main(int argc, char** argv){

    const char* method = "yolo";
    if(argc > 1){
        method = argv[1];
    }

    if(strcmp(method, "yolo") == 0){
        app_yolo();
    }else if(strcmp(method, "alphapose") == 0){
        app_alphapose();
    }else if(strcmp(method, "fall_recognize") == 0){
        app_fall_recognize();
    }else if(strcmp(method, "retinaface") == 0){
        app_retinaface();
    }else if(strcmp(method, "arcface") == 0){
        app_arcface();
    }else if(strcmp(method, "arcface_video") == 0){
        app_arcface_video();
    }else if(strcmp(method, "arcface_tracker") == 0){
        app_arcface_tracker();
    }else{
        printf(
            "Help: \n"
            "    ./pro method[yolo、alphapose、fall_recognize、retinaface、arcface、arcface_video、arcface_tracker]\n"
            "\n"
            "    ./pro yolo\n"
            "    ./pro alphapose\n"
            "    ./pro fall_recognize\n"
        );
    }
    return 0;
}