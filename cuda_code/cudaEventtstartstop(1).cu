cudaEvent_t start, stop;

cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, 0);

//work kernel

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float worktime;

cudaEventElapsedTime(&worktime, start, stop);

printf("Time = %3.1f ms \n", worktime);
cudaEventDestroy(start);
cudaEventDestroy(stop);