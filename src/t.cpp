#include <stdio.h>

int main111(){

    const int a = ('A', 'a');
    printf("%c", ('A', 'a'));

    char s[] = "Beijing ligong daxue";
    int i, j;
    for(i = j = 0; s[i] != '\0'; i++){
        if(s[i] != ' '){
            s[j++] = s[i];
        }else{
        }
    }
    s[j] = '\0';
    printf("%s", s);
    return 0;
}

int main22(){

    long num = 0;
    FILE* fp = NULL;

    if((fp = fopen("fname.dat", "r")) == nullptr){
        printf("Can't open the file! ");
        //exit(0);
    }

    while(!feof(fp)){
        fgetc(fp);
        num++;
    }

    printf("num=%d\n", num);
    fclose(fp);
}

void revstr(char* s){

    char* p = s, c;
    while(*p) p++;
    p--;

    if(s < p){
        c = *s;
        *s = *p;
        *p ='\0';
        revstr(s+1);
        *p = c;
    }
}

int main123123(){

    char t[] = "abcde";
    revstr(t);
    printf("%s\n", t);
    return 0;
}

#include <stdio.h>

// int pow3(int n, int x){

//     int i, last;
//     for(last = 1, i = 1, i <= x; i++)
//         last = /**/;
    
//     return last;
// }

// int main(){

//     int x, n, min, flag=1;
//     scanf("%d", &n);

//     for(min = 2; flag; min++)
//         for(x=1; x < min && flag; x++)
//             if(/**/ && pow3(n, x) == pow3(n, min-x)){
//                 printf("x=%d,y=%d", x, min-x);
//                 /**/
//             }
//     return 0;
// }

// /******************************************************************************

//                             Online C Compiler.
//                 Code, Compile, Run and Debug C program online.
// Write your code in this editor and press "Run" button to compile and execute it.

// *******************************************************************************/

#include <stdio.h>
int pow3(int n, int x){
    int i, last;
    for (last = 1, i = 1; i <= x ; i++)
        last = last * n;
    return (last);
}
    
int main(){
    int x,n,min, flag = 1;
    scanf("%d",&n);
    for(min = 2; flag; min++)
        for (x = 1; x< min&&flag; x++)
            if ((x != min - x) && pow3(n, x) == pow3(n, min - x)){
                printf("x=%d,y = %d",x,min- x);
                flag = 0;
            }
    return 0;        
}