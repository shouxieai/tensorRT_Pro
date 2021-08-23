
#include <stdio.h>
#include <string.h>

int yolo_main();
int alphapose_main();
int app_fall_recognize();

int main(int argc, char** argv){

    const char* method = "yolo";
    if(argc > 1){
        method = argv[1];
    }

    if(strcmp(method, "yolo") == 0){
        yolo_main();
    }else if(strcmp(method, "alphapose") == 0){
        alphapose_main();
    }else if(strcmp(method, "app_fall_recognize") == 0){
        app_fall_recognize();
    }else{
        printf(
            "Help: \n"
            "    ./pro method[yolo、alphapose、app_fall_recognize]\n"
            "\n"
            "    ./pro yolo\n"
            "    ./pro alphapose\n"
            "    ./pro app_fall_recognize\n"
        );
    }
    return 0;
}