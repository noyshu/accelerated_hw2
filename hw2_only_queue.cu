/* compile with: nvcc -O3 -maxrregcount=32 hw2.cu -o hw2 */

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <algorithm>

#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000
#define NREQUESTS 100000
/*#define NREQUESTS 3*/

#define NUM_OF_STREAMS 64
/*#define NUM_OF_STREAMS 4*/
#define NOT_A_VALID_JOB (-1)
#define NO_FREE_STREAM (-2)
typedef unsigned char uchar;
#define OUT

#define SIZE 10
#define ERROR (-10)
#define REGS_PER_THREAD 32
#define HISTOGRAM_DIM 256
#define JOB_NOT_FOUND -2
#define BLOCK_SHARED_MEM ((sizeof(double) + sizeof(int))*(2*HISTOGRAM_DIM + 1))

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}

/* we'll use these to rate limit the request load */
struct rate_limit_t {
    double last_checked;
    double lambda;
    unsigned seed;
};

void rate_limit_init(struct rate_limit_t *rate_limit, double lambda, int seed) {
    rate_limit->lambda = lambda;
    rate_limit->seed = (seed == -1) ? 0 : seed;
    rate_limit->last_checked = 0;
}

int rate_limit_can_send(struct rate_limit_t *rate_limit) {
    if (rate_limit->lambda == 0) return 1;
    double now = get_time_msec() * 1e-3;
    double dt = now - rate_limit->last_checked;
    double p = dt * rate_limit->lambda;
    rate_limit->last_checked = now;
    if (p > 1) p = 1;
    double r = (double)rand_r(&rate_limit->seed) / RAND_MAX;
    return (p > r);
}

void rate_limit_wait(struct rate_limit_t *rate_limit) {
    while (!rate_limit_can_send(rate_limit)) {
        usleep(1. / (rate_limit->lambda * 1e-6) * 0.01);
    }
}

/* we won't load actual files. just fill the images with random bytes */
void load_image_pairs(uchar *images1, uchar *images2) {
    srand(0);
    for (int i = 0; i < N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION; i++) {
        images1[i] = rand() % 256;
        images2[i] = rand() % 256;
    }
}

__device__ __host__ bool is_in_image_bounds(int i, int j) {
    return (i >= 0) && (i < IMG_DIMENSION) && (j >= 0) && (j < IMG_DIMENSION);
}

__device__ __host__ uchar local_binary_pattern(uchar *image, int i, int j) {
    uchar center = image[i * IMG_DIMENSION + j];
    uchar pattern = 0;
    if (is_in_image_bounds(i - 1, j - 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j - 1)] >= center) << 7;
    if (is_in_image_bounds(i - 1, j    )) pattern |= (image[(i - 1) * IMG_DIMENSION + (j    )] >= center) << 6;
    if (is_in_image_bounds(i - 1, j + 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j + 1)] >= center) << 5;
    if (is_in_image_bounds(i    , j + 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j + 1)] >= center) << 4;
    if (is_in_image_bounds(i + 1, j + 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j + 1)] >= center) << 3;
    if (is_in_image_bounds(i + 1, j    )) pattern |= (image[(i + 1) * IMG_DIMENSION + (j    )] >= center) << 2;
    if (is_in_image_bounds(i + 1, j - 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j - 1)] >= center) << 1;
    if (is_in_image_bounds(i    , j - 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j - 1)] >= center) << 0;
    return pattern;
}

void image_to_histogram(uchar *image, int *histogram) {
    memset(histogram, 0, sizeof(int) * 256);
    for (int i = 0; i < IMG_DIMENSION; i++) {
        for (int j = 0; j < IMG_DIMENSION; j++) {
            uchar pattern = local_binary_pattern(image, i, j);
            histogram[pattern]++;
        }
    }
}

double histogram_distance(int *h1, int *h2) {
    /* we'll use the chi-square distance */
    double distance = 0;
    for (int i = 0; i < 256; i++) {
        if (h1[i] + h2[i] != 0) {
            distance += ((double)SQR(h1[i] - h2[i])) / (h1[i] + h2[i]);
        }
    }
    return distance;
}

__global__ void gpu_image_to_histogram(uchar *image, int *histogram) {
    uchar pattern = local_binary_pattern(image, threadIdx.x / IMG_DIMENSION, threadIdx.x % IMG_DIMENSION);
    atomicAdd(&histogram[pattern], 1);
}

__global__ void gpu_histogram_distance(int *h1, int *h2, double *distance) {
    int length = 256;
    int tid = threadIdx.x;
    distance[tid] = 0;
    if (h1[tid] + h2[tid] != 0) {
        distance[tid] = ((double)SQR(h1[tid] - h2[tid])) / (h1[tid] + h2[tid]);
    }
    __syncthreads();

    while (length > 1) {
        if (threadIdx.x < length / 2) {
            distance[tid] = distance[tid] + distance[tid + length / 2];
        }
        length /= 2;
        __syncthreads();
    }
}

void print_usage_and_die(char *progname) {
    printf("usage:\n");
    printf("%s streams <load (requests/sec)>\n", progname);
    printf("OR\n");
    printf("%s queue <#threads> <load (requests/sec)>\n", progname);
    exit(1);
}

void startJob(cudaStream_t stream, uchar* cpu_image1, uchar* cpu_image2,
		int* gpu_hist1, int* gpu_hist2, double* gpu_hist_distance,
		uchar* gpu_image1, uchar* gpu_image2) {
	//TODO allocate zone, can be extracted from here:

	CUDA_CHECK(cudaMemcpyAsync(gpu_image1, cpu_image1, IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(gpu_image2, cpu_image2, IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice));
	cudaMemset(gpu_hist1, 0, 256 * sizeof(int));
	cudaMemset(gpu_hist2, 0, 256 * sizeof(int));
	gpu_image_to_histogram <<<1, 1024, 0, stream>>> (gpu_image1, gpu_hist1);
	gpu_image_to_histogram <<<1, 1024, 0, stream>>> (gpu_image2, gpu_hist2);
	gpu_histogram_distance <<<1, 256, 0, stream>>> (gpu_hist1, gpu_hist2, gpu_hist_distance);//TODO check that actually needs to be a stream
}

enum {PROGRAM_MODE_STREAMS = 0, PROGRAM_MODE_QUEUE};

void initStreams(cudaStream_t* streams, int* gpu_hists1[NUM_OF_STREAMS], int* gpu_hists2[NUM_OF_STREAMS], double* gpu_distances[NUM_OF_STREAMS], uchar* gpu_imgs1[NUM_OF_STREAMS], uchar* gpu_imgs2[NUM_OF_STREAMS])
{
	for (int i = 0; i < NUM_OF_STREAMS; i++) {
		CUDA_CHECK(cudaMalloc(&gpu_imgs1[i], IMG_DIMENSION * IMG_DIMENSION * sizeof(uchar)));
		CUDA_CHECK(cudaMalloc(&gpu_imgs2[i], IMG_DIMENSION * IMG_DIMENSION * sizeof(uchar)));
		CUDA_CHECK(cudaMalloc(&gpu_hists1[i], 256 * sizeof(int)));
		CUDA_CHECK(cudaMalloc(&gpu_hists2[i], 256 * sizeof(int)));
		CUDA_CHECK(cudaMalloc(&gpu_distances[i], 256 * sizeof(double)));
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}
}

void cleanStreams(cudaStream_t* streams,int* gpu_hists1[NUM_OF_STREAMS],int* gpu_hists2[NUM_OF_STREAMS],double* gpu_distances[NUM_OF_STREAMS],uchar* gpu_imgs1[NUM_OF_STREAMS],uchar* gpu_imgs2[NUM_OF_STREAMS])
{
	for (int i = 0; i < NUM_OF_STREAMS; i++) {
		CUDA_CHECK(cudaFree(gpu_imgs1[i]));
		CUDA_CHECK(cudaFree(gpu_imgs2[i]));
		CUDA_CHECK(cudaFree(gpu_hists1[i]));
		CUDA_CHECK(cudaFree(gpu_hists2[i]));
		CUDA_CHECK(cudaFree(gpu_distances[i]));
		CUDA_CHECK(cudaStreamDestroy(streams[i]));
	}
}

void checkForCompletedRequests(cudaStream_t* streams, int* currentJob, double* req_t_end, double* total_distance, double** gpu_hist_distance)
{
	for(int i = 0; i < NUM_OF_STREAMS; i++)
	{
		cudaError_t res = cudaStreamQuery(streams[i]);
		if(res == cudaSuccess && currentJob[i] != NOT_A_VALID_JOB)
		{
			double cpu_hist_distance;
			cudaMemcpy(&cpu_hist_distance, gpu_hist_distance[i], sizeof(double), cudaMemcpyDeviceToHost);
			/*printf("checkFor: dist[%d] = %f \n", currentJob[i], cpu_hist_distance);//TODO*/
			*total_distance += cpu_hist_distance;
			req_t_end[currentJob[i]] = get_time_msec();
			currentJob[i] = NOT_A_VALID_JOB;
		} else if (res == cudaErrorNotReady) {
			continue;//Not free yet
		} else {
			CUDA_CHECK(res);//Error result
		}
	}
}

int findFreeStream(cudaStream_t* streams, int* currentJob) {
	for (int i = 0; i < NUM_OF_STREAMS; i++) {
		if(currentJob[i] == NOT_A_VALID_JOB)
			return i;
	}
	return NO_FREE_STREAM;
}

void waitForCompletedRequests(cudaStream_t* streams, int* currentJob, double* req_t_end, double* total_distance, double** gpu_hist_distance)
{
	bool flag = true;
	while(flag)
	{
		flag = false;
		for(int i = 0; i < NUM_OF_STREAMS; i++)
		{
			if(currentJob[i] != NOT_A_VALID_JOB) {//This streams job is valid
				flag = true;
			} else {
				continue;
			}
			cudaError_t res = cudaStreamQuery(streams[i]);
			if(res == cudaSuccess)
			{
				double cpu_hist_distance;
				cudaMemcpy(&cpu_hist_distance, gpu_hist_distance[i], sizeof(double), cudaMemcpyDeviceToHost);
				*total_distance += cpu_hist_distance;
				req_t_end[currentJob[i]] = get_time_msec();
				currentJob[i] = NOT_A_VALID_JOB;
			} else if (res == cudaErrorNotReady) {
				continue;
			} else {
				CUDA_CHECK(res);
			}
		}	
	}
}

int getNumOfThreadBlocks(int threadsPerBlock) {
	int threadRegs = REGS_PER_THREAD;
	int threadBlockRegs = threadRegs*threadsPerBlock;
	int sum = 0;
	int NumOfDevices = ERROR;
	cudaGetDeviceCount(&NumOfDevices);
	//TODO assert:
	if(NumOfDevices == ERROR) exit(ERROR);

	int multiProcessorCount, maxThreadsPerMultiProcessor, regsPerMultiprocessor;
	int threadBlocksPerDevice, smRegsBlocks, smThreadsBlocks, sharedMemBlocks;
	int blocksInSm;
	size_t sharedMemPerMultiprocessor;
	cudaDeviceProp prop;
	for (int i = 0; i < NumOfDevices; i++) {
		cudaGetDeviceProperties(&prop, i);
		multiProcessorCount = prop.multiProcessorCount;
		maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
		sharedMemPerMultiprocessor = prop.sharedMemPerMultiprocessor;
		regsPerMultiprocessor = prop.regsPerMultiprocessor;

		smRegsBlocks = regsPerMultiprocessor / threadBlockRegs;
		smThreadsBlocks = maxThreadsPerMultiProcessor / threadsPerBlock;
		sharedMemBlocks = sharedMemPerMultiprocessor / BLOCK_SHARED_MEM;

		blocksInSm = std::min(smRegsBlocks, std::min(smThreadsBlocks, sharedMemBlocks));

		threadBlocksPerDevice = blocksInSm * multiProcessorCount;
		sum += threadBlocksPerDevice;
	}
	//TODO assert
	if (sum == 0) exit(ERROR);
	return sum;
}

class cpu2gpuQueue
{
private:
	volatile int head;
	volatile int tail;
	volatile int maxHead;
	volatile int curJob[SIZE];
public:
	cpu2gpuQueue(){
		head = 0;
		tail = 0;
		maxHead = SIZE;
	}
	__device__ int consume()
	{
		if(tail<head)
		{
			int job = (int)curJob[tail%SIZE];
			__threadfence_system();
			tail++;
			maxHead++;
			__threadfence_system();
			return job;
		}
		return JOB_NOT_FOUND;
	}
	__device__ int consume_block()
	{
		int jobIdx = JOB_NOT_FOUND;
		while(jobIdx == JOB_NOT_FOUND)
		{
			jobIdx = consume();
		}
		return jobIdx;
	}
	__host__ bool produce(int jobIdx)
	{
		if(maxHead == head)
			return false;
		curJob[head%SIZE] = jobIdx;
		head++;
		return true;
	}
	__host__ void produce_block(int jobIdx)
	{
		while(!produce(jobIdx));
	}
};

class gpu2cpuQueue
{
private:
	volatile int head;
	volatile int tail;
	volatile int maxHead;
	volatile int curJob[SIZE];
	volatile double queueDist[SIZE];
public:
	gpu2cpuQueue(){
		head = 0;
		tail = 0;
		maxHead = SIZE;
	}
	__host__ bool consume(double* distance,int* jobIdx)
	{
		if(tail<head)
		{
			*distance = (double)queueDist[tail%SIZE];
			*jobIdx = (int)curJob[tail%SIZE];
			tail++;
			maxHead++;
			return true;
		}
		return false;
	}
	__host__ void consume_block(double* distance,int* jobIdx)
	{
		while(!consume(distance,jobIdx));
	}
	__device__ bool produce(double distance,int jobIdx)
	{
		if(maxHead == head)
			return false;
		queueDist[head%SIZE] = distance;
		curJob[head%SIZE] = jobIdx;
		__threadfence_system();
		head++;
		__threadfence_system();//TODO check if actually improves performance
		return true;
	}
	__device__ void produce_block(double distance,int jobIdx)
	{
		while(!produce(distance,jobIdx)) ;
	}
};

__device__ void clearArrays(int* hist1,int* hist2)
{
	for(int i=threadIdx.x;i<HISTOGRAM_DIM;i+=blockDim.x)
	{
		hist1[i]=0;
		hist2[i]=0;
	}
	__syncthreads();
}


__device__ void copyImg(uchar* image1, uchar* image2, uchar* g_image1, uchar* g_image2){
	for(int i=threadIdx.x; i< IMG_DIMENSION*IMG_DIMENSION; i+=blockDim.x)
	{
		g_image1[i] = image1[i];
		g_image2[i] = image2[i];
	}
	__syncthreads();
}

__device__ void calcHist(uchar* image, int* hist){
	for(int i=threadIdx.x; i< IMG_DIMENSION*IMG_DIMENSION; i+=blockDim.x)
	{
		uchar pattern = local_binary_pattern(image, i / IMG_DIMENSION, i % IMG_DIMENSION);
		atomicAdd(&hist[pattern], 1);
	}
}


__device__ void calcDist(int *h1, int *h2, double *distance) 
{
    int length = HISTOGRAM_DIM;
	for(int i= threadIdx.x ; i<length ; i+=blockDim.x)
	{
		distance[i] = 0;
		if (h1[i] + h2[i] != 0) {
			distance[i] = ((double)SQR(h1[i] - h2[i])) / (h1[i] + h2[i]);
		}
    }
    __syncthreads();

    while (length > 1) {
		for(int i=threadIdx.x;i<length/2;i+=blockDim.x)
            distance[i] = distance[i] + distance[i + length / 2];
        length /= 2;
        __syncthreads();
    }
	__syncthreads();
}

__global__ void blockAlgo(gpu2cpuQueue** gpu2cpus, cpu2gpuQueue** cpu2gpus,uchar* image1,uchar* image2)
{
	gpu2cpuQueue* gpu2cpu = gpu2cpus[blockIdx.x];
	cpu2gpuQueue* cpu2gpu = cpu2gpus[blockIdx.x];
	__shared__ uchar s_image1[IMG_DIMENSION*IMG_DIMENSION];
	__shared__ uchar s_image2[IMG_DIMENSION*IMG_DIMENSION];
	__shared__ int s_hist1[HISTOGRAM_DIM];
	__shared__ int s_hist2[HISTOGRAM_DIM];
	__shared__ double s_dist[HISTOGRAM_DIM];
	while(1)
	{
		__shared__ int jobIdx;
		clearArrays(s_hist1,s_hist2);
		
		if(!threadIdx.x) 
			jobIdx = cpu2gpu->consume_block();
		__syncthreads();
		if(jobIdx == NOT_A_VALID_JOB)
		{
			break;
		}
		int num = (jobIdx%N_IMG_PAIRS) * IMG_DIMENSION * IMG_DIMENSION;
		copyImg(&image1[num],&image2[num],s_image1,s_image2);
		calcHist(s_image1,s_hist1);
		calcHist(s_image2,s_hist2);
		__syncthreads();
		calcDist(s_hist1,s_hist2,s_dist);
		if(!threadIdx.x) gpu2cpu->produce_block(s_dist[0],jobIdx);
		__syncthreads();
	}
}

void CpuConsume(gpu2cpuQueue** gpu2cpus,int numOfBlocks,double* req_t_end,int* remainJobs,double* total_distance)
{
	for(int i=0;i<numOfBlocks;i++)
	{
		double distance;
		int jobIdx;
		
		if(gpu2cpus[i]->consume(&distance,&jobIdx))
		{
			req_t_end[jobIdx] = get_time_msec(); 
			remainJobs[0]--;
			total_distance[0] += distance;
		}
	}
}
bool CpuProduce(cpu2gpuQueue** cpu2gpus,int numOfBlocks,int jobID,int* lastProduced)
{
	for(int i=*lastProduced;i<numOfBlocks+*lastProduced;i++)
	{
		if(cpu2gpus[i%numOfBlocks]->produce(jobID))
		{
			*lastProduced = (i+1)%numOfBlocks;
			return true;
		}
	}
	return false;
}

void TryPushEndEvent(cpu2gpuQueue** cpu2gpus,int numOfBlocks,int* pushed,int* remainPush)
{
	for(int i=0;i<numOfBlocks;i++)
	{
		if(pushed[i])
		{
			continue;
		}
		if(cpu2gpus[i]->produce(NOT_A_VALID_JOB))
		{
			pushed[i] = 1;
			remainPush[0] --;
		}
	}

}


int main(int argc, char *argv[]) {

    int mode = -1;
    int threads_queue_mode = -1; /* valid only when mode = queue */
    double load = 0;
    if (argc < 3) print_usage_and_die(argv[0]);

    if        (!strcmp(argv[1], "streams")) {
        if (argc != 3) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_STREAMS;
        load = atof(argv[2]);
    } else if (!strcmp(argv[1], "queue")) {
        if (argc != 4) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_QUEUE;
        threads_queue_mode = atoi(argv[2]);
        load = atof(argv[3]);
    } else {
        print_usage_and_die(argv[0]);
    }

    uchar *images1,*gpu_images1; /* we concatenate all images in one huge array */
    uchar *images2,*gpu_images2;
    CUDA_CHECK( cudaHostAlloc(&images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
    CUDA_CHECK( cudaHostAlloc(&images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
	
    load_image_pairs(images1, images2);
    double t_start, t_finish;
    double total_distance = 0;
	/* using CPU */
    printf("\n=== CPU ===\n");
    int histogram1[256];
    int histogram2[256];
    t_start  = get_time_msec();
    for (int i = 0; i < NREQUESTS; i++) {
        int img_idx = i % N_IMG_PAIRS;
        image_to_histogram(&images1[img_idx * IMG_DIMENSION * IMG_DIMENSION], histogram1);
        image_to_histogram(&images2[img_idx * IMG_DIMENSION * IMG_DIMENSION], histogram2);
        total_distance += histogram_distance(histogram1, histogram2);
    }
    t_finish = get_time_msec();
    printf("average distance between images %f\n", total_distance / NREQUESTS);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);
	
    printf("\n=== Client-Server ===\n");
    total_distance = 0;
    double *req_t_start = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_start, 0, NREQUESTS * sizeof(double));

    double *req_t_end = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_end, 0, NREQUESTS * sizeof(double));

    struct rate_limit_t rate_limit;
    rate_limit_init(&rate_limit, load, 0);
	double ti = get_time_msec();	
	
	if (mode == PROGRAM_MODE_STREAMS) {
		
    } else if (mode == PROGRAM_MODE_QUEUE) {
		int threadBlocks = getNumOfThreadBlocks(threads_queue_mode);
		printf("threadBlocks = %d",threadBlocks);
		gpu2cpuQueue **cpu_gpu2cpus, **gpu_gpu2cpus;
		cpu2gpuQueue **cpu_cpu2gpus, **gpu_cpu2gpus;
		
		
	
		CUDA_CHECK( cudaMalloc(&gpu_images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION));
		CUDA_CHECK( cudaMalloc(&gpu_images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION));
		CUDA_CHECK(cudaMemcpy(gpu_images1, images1, N_IMG_PAIRS*IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(gpu_images2, images2, N_IMG_PAIRS*IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice));
		
    	CUDA_CHECK(cudaHostAlloc(&cpu_gpu2cpus, threadBlocks*sizeof(gpu2cpuQueue*), 0));
		CUDA_CHECK(cudaHostAlloc(&cpu_cpu2gpus, threadBlocks*sizeof(cpu2gpuQueue*), 0));
		
		CUDA_CHECK( cudaMalloc( &gpu_gpu2cpus,threadBlocks*sizeof(gpu2cpuQueue*)));
		
		CUDA_CHECK( cudaMalloc( &gpu_cpu2gpus,threadBlocks*sizeof(cpu2gpuQueue*)) );
		
		gpu2cpuQueue* cpu_gpu_gpu2cpu[threadBlocks];
		cpu2gpuQueue* cpu_gpu_cpu2cpu[threadBlocks];
		
		for(int i=0;i<threadBlocks;i++)
		{
			CUDA_CHECK(cudaHostAlloc(&cpu_gpu2cpus[i], sizeof(gpu2cpuQueue), 0));
			CUDA_CHECK(cudaHostAlloc(&cpu_cpu2gpus[i],sizeof(cpu2gpuQueue), 0));
			*(cpu_gpu2cpus[i]) = gpu2cpuQueue();
			*(cpu_cpu2gpus[i]) = cpu2gpuQueue();
			CUDA_CHECK( cudaHostGetDevicePointer(&cpu_gpu_gpu2cpu[i],cpu_gpu2cpus[i],0) );
			CUDA_CHECK( cudaHostGetDevicePointer(&cpu_gpu_cpu2cpu[i],cpu_cpu2gpus[i],0) );
		}
		CUDA_CHECK(cudaMemcpy(gpu_gpu2cpus, cpu_gpu_gpu2cpu, threadBlocks*sizeof(gpu2cpuQueue*), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(gpu_cpu2gpus, cpu_gpu_cpu2cpu, threadBlocks*sizeof(cpu2gpuQueue*), cudaMemcpyHostToDevice));
		

		blockAlgo<<<threadBlocks, threads_queue_mode>>>(gpu_gpu2cpus,gpu_cpu2gpus,gpu_images1,gpu_images2);

		int remainJobs = NREQUESTS;
		int lastProduced = 0;
        for (int i = 0; i < NREQUESTS; i++) 
		{
			CpuConsume(cpu_gpu2cpus,threadBlocks,req_t_end,&remainJobs,&total_distance);
			rate_limit_wait(&rate_limit);
            int img_idx = i % N_IMG_PAIRS;
			fflush(stdout);
			if(CpuProduce(cpu_cpu2gpus,threadBlocks,i,&lastProduced))
			{
				req_t_start[i] = get_time_msec();
				continue;
			}
			i--;
        }
		int* pushed = (int*)malloc(threadBlocks*sizeof(int));
		int remainPushes = threadBlocks;
		for(int i=0;i<threadBlocks;i++) pushed[i] = 0;
		while(remainJobs || remainPushes)
		{
			CpuConsume(cpu_gpu2cpus,threadBlocks,req_t_end,&remainJobs,&total_distance);
			TryPushEndEvent(cpu_cpu2gpus,threadBlocks,pushed,&remainPushes);
		}
		cudaDeviceSynchronize();
		for(int i=0;i<threadBlocks;i++)
		{
			CUDA_CHECK(cudaFreeHost(cpu_gpu2cpus[i]));
			CUDA_CHECK(cudaFreeHost(cpu_cpu2gpus[i]));
		}
		CUDA_CHECK(cudaFreeHost(cpu_gpu2cpus));
		CUDA_CHECK(cudaFreeHost(cpu_cpu2gpus));
    } else {
        assert(0);
    }
    double tf = get_time_msec();

    double avg_latency = 0;
    for (int i = 0; i < NREQUESTS; i++) {
        avg_latency += (req_t_end[i] - req_t_start[i]);
    }
    avg_latency /= NREQUESTS;


    printf("mode = %s\n", mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %lf (req/sec)\n", load);
    if (mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", threads_queue_mode);
    printf("average distance between images %f\n", total_distance / NREQUESTS);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
    printf("average latency = %lf (msec)\n", avg_latency);

	
	free(req_t_end);
	free(req_t_start);
	CUDA_CHECK( cudaFreeHost(images1));
    CUDA_CHECK( cudaFreeHost(images2));
    return 0;
}
