
#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>

int app_yolo();
int app_yolo_gpuptr();
int app_alphapose();
int app_fall_recognize();
int app_retinaface();
int app_arcface();
int app_arcface_video();
int app_arcface_tracker();
int app_scrfd();
int app_high_performance();
int app_lesson();
int app_plugin();    
int app_yolo_fast();
int app_centernet();
int app_dbface();
int app_bert();
int direct_yolo();
int direct_unet();
int direct_mae();
int direct_classifier();
int test_warpaffine();
int test_yolo_map();

int main(int argc, char** argv){
    
    const char* method = "yolo";
    if(argc > 1){
        method = argv[1];
    }

    if(strcmp(method, "yolo") == 0){
        app_yolo();
    }else if(strcmp(method, "yolo_gpuptr") == 0){
        app_yolo_gpuptr();
    }else if(strcmp(method, "yolo_fast") == 0){
        app_yolo_fast();
    }else if(strcmp(method, "dyolo") == 0){
        direct_yolo();
    }else if(strcmp(method, "dunet") == 0){
        direct_unet();
    }else if(strcmp(method, "dmae") == 0){
        direct_mae();
    }else if(strcmp(method, "dclassifier") == 0){
        direct_classifier();
    }else if(strcmp(method, "bert") == 0){
        app_bert();
    }else if(strcmp(method, "centernet") == 0){
        app_centernet();
    }else if(strcmp(method, "dbface") == 0){
        app_dbface();
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
    }else if(strcmp(method, "scrfd") == 0){
        app_scrfd();
    }else if(strcmp(method, "test_warpaffine") == 0){
        test_warpaffine();
    }else if(strcmp(method, "test_yolo_map") == 0){
        test_yolo_map();
    }else if(strcmp(method, "high_perf") == 0){
        app_high_performance();
    }else if(strcmp(method, "lesson") == 0){
        app_lesson();
    }else if(strcmp(method, "plugin") == 0){
        app_plugin();
    }else{
        printf("Unknow method: %s\n", method);
        printf(
            "Help: \n"
            "    ./pro method[yolo、alphapose、fall、retinaface、arcface、arcface_video、arcface_tracker]\n"
            "\n"
            "    ./pro yolo\n"
            "    ./pro alphapose\n"
            "    ./pro fall\n"
        );
    } 
    return 0;
}
