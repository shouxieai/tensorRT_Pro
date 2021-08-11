
#include <stdio.h>
#include <string.h>

int yolov5_main();
int yolox_main();

int main(int argc, char** argv){

    const char* method = "yolov5";
    if(argc > 1){
        method = argv[1];
    }

    if(strcmp(method, "yolov5") == 0){
        yolov5_main();
    }else if(strcmp(method, "yolox") == 0){
        yolox_main();
    }else{
        printf(
            "Help: \n"
            "    ./pro method[yolov5 or yolox]\n"
            "\n"
            "    ./pro yolov5\n"
            "    ./pro yolox\n"
        );
    }
    return 0;
}