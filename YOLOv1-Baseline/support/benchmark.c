/* Measurement-only extension. Training/inference kernels remain upstream code. */
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "network.h"
#include "parser.h"
#include "cuda.h"
#include "box.h"
extern void convert_yolo_detections(float*,int,int,int,int,int,int,float,float**,box*,int);

static double wall_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void synchronize_gpu(void) {
#ifdef GPU
    if(gpu_index >= 0) check_error(cudaDeviceSynchronize());
#endif
}
static int compare_double(const void *a,const void *b) {
    double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);
}
void run_benchmark(int argc,char **argv) {
    if(argc != 7) {
        fprintf(stderr,"benchmark cfg weights image iterations output.json\n"); exit(2);
    }
    int count=atoi(argv[5]),warmup=20,i,j;
    if(count<1) exit(2);
    network net=parse_network_cfg(argv[2]);
    load_weights(&net,argv[3]);
    set_batch_network(&net,1);
    image im=load_image_color(argv[4],0,0);
    image sized=resize_image(im,net.w,net.h);
    layer l=net.layers[net.n-1];
    int total=l.side*l.side*l.n;
    box *boxes=calloc(total,sizeof(box));
    float **probs=calloc(total,sizeof(float*));
    for(j=0;j<total;++j) probs[j]=calloc(l.classes,sizeof(float));
    for(i=0;i<warmup;++i) network_predict(net,sized.data);
    synchronize_gpu();
    double *durations=calloc(count,sizeof(double)),sum=0,pipeline=0;
    for(i=0;i<count;++i) {
        synchronize_gpu(); double start=wall_seconds();
        network_predict(net,sized.data);
        synchronize_gpu(); durations[i]=wall_seconds()-start;sum+=durations[i];
    }
    for(i=0;i<count;++i) {
        synchronize_gpu(); double start=wall_seconds();
        image original=load_image_color(argv[4],0,0);
        image resized=resize_image(original,net.w,net.h);
        float *pred=network_predict(net,resized.data);
        convert_yolo_detections(pred,l.classes,l.n,l.sqrt,l.side,original.w,original.h,.001,probs,boxes,0);
        do_nms_sort(boxes,probs,total,l.classes,.5);
        free_image(original); free_image(resized);
        synchronize_gpu();pipeline+=wall_seconds()-start;
    }
    qsort(durations,count,sizeof(double),compare_double);
    FILE *fp=fopen(argv[6],"w");if(!fp) {perror(argv[6]);exit(2);}
    fprintf(fp,"{\"batch\":1,\"precision\":\"FP32\",\"warmup\":%d,\"iterations\":%d,\"gpu_index\":%d,",warmup,count,gpu_index);
    fprintf(fp,"\"network_api_fps\":%.8f,\"pipeline_fps\":%.8f,\"mean_ms\":%.8f,\"p50_ms\":%.8f,\"p95_ms\":%.8f,",count/sum,count/pipeline,1000*sum/count,1000*durations[(count-1)/2],1000*durations[(int)((count-1)*.95)]);
    fprintf(fp,"\"network_scope\":\"network_predict API including its H2D/D2H; CUDA synchronized\",\"pipeline_scope\":\"cached-file read, decode, original resize, network API, original decode and NMS; excludes drawing\",\"score_threshold\":0.001,\"nms_iou\":0.5}");
    fclose(fp);
    free(durations);for(j=0;j<total;++j)free(probs[j]);free(probs);free(boxes);
    free_image(sized);free_image(im);free_network(net);
}
