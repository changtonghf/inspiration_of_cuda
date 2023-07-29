#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>

const uint crc32df[256] = {
    0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA,
    0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
    0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988,
    0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91,
    0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE,
    0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
    0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC,
    0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5,
    0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
    0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
    0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940,
    0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
    0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116,
    0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
    0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
    0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
    0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A,
    0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
    0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818,
    0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01,
    0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
    0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457,
    0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C,
    0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
    0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2,
    0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB,
    0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
    0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
    0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086,
    0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
    0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4,
    0x59B33D17, 0x2EB40D81, 0xB7BD5C3B, 0xC0BA6CAD,
    0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A,
    0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683,
    0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8,
    0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
    0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE,
    0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7,
    0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC,
    0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5,
    0xD6D6A3E8, 0xA1D1937E, 0x38D8C2C4, 0x4FDFF252,
    0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
    0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60,
    0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79,
    0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236,
    0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F,
    0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04,
    0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
    0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A,
    0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713,
    0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38,
    0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21,
    0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E,
    0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
    0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C,
    0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
    0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2,
    0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB,
    0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0,
    0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
    0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6,
    0xBAD03605, 0xCDD70693, 0x54DE5729, 0x23D967BF,
    0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
    0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
};

typedef struct
{
    uint    cid;
    uint    mat;
} dti;

typedef struct
{
    uint    nid;
    float   xid;
    float   yid;
    float   zid;
} kti;

typedef struct
{
    uint    hash;
    uint    next;
    uint    rowid;
    uint    __padding__;
    kti     t;
} khi;

typedef struct {
    uint    crc32t[256];
    uint    length;
    uint    ncols;
    uint    nitems;
    uint    nslots;
    uint    hash_min;
    uint    hash_max;
} kds;

#define get_local_id()      (threadIdx.x)
#define get_local_size()    (blockDim.x)
#define get_global_id()     (threadIdx.x + blockIdx.x * blockDim.x)
#define get_global_size()   (blockDim.x * gridDim.x)
#define get_global_base()   (blockIdx.x * blockDim.x)
#define get_global_index()  (blockIdx.x)

#define TYPEALIGN(ALIGNVAL,LEN) (((uint) (LEN) + ((ALIGNVAL) - 1)) & ~((uint) ((ALIGNVAL) - 1)))

#define INIT_LEGACY_CRC32(crc) ((crc)  = 0xFFFFFFFF)
#define FIN_LEGACY_CRC32(crc)  ((crc) ^= 0xFFFFFFFF)
#define COMP_LEGACY_CRC32(crc, data, len) COMP_CRC32_REFLECTED_TABLE(crc, data, len, crc32df)
#define EQ_LEGACY_CRC32(c1, c2) ((c1) == (c2))

#define COMP_CRC32_REFLECTED_TABLE(crc, data, len, table) \
do { \
    const unsigned char *__data = (const unsigned char *) (data); \
    uint __len = (len); \
    while (__len-- > 0) \
    { \
        int __tab_index = ((int) ((crc) >> 24) ^ *__data++) & 0xFF; \
        (crc) = table[__tab_index] ^ ((crc) << 8); \
    } \
} while (0)

#define KERN_DATA_STORE_HEAD_LENGTH(kds) TYPEALIGN(16,sizeof(kds))
#define KERN_DATA_STORE_BODY(kds) ((char *)(kds) + KERN_DATA_STORE_HEAD_LENGTH(kds))
#define KERN_DATA_STORE_ROWINDEX(kds) ((uint *)(KERN_DATA_STORE_BODY(kds)))
#define KERN_DATA_STORE_HASHSLOT(kds) ((uint *)(KERN_DATA_STORE_BODY(kds) + TYPEALIGN(16,100000 * sizeof(uint))))

static inline uint HOST_HASH_VALUE(int key)
{
    uint hash;
    INIT_LEGACY_CRC32(hash);
    COMP_LEGACY_CRC32(hash, &key, sizeof(int));
    FIN_LEGACY_CRC32(hash);
    return hash;
}

__device__ __forceinline__ static khi* KERN_HASH_FIRST_ITEM(kds *k, uint h)
{
    uint *slot = (uint *)(((char *)(k) + KERN_DATA_STORE_HEAD_LENGTH(kds)) + TYPEALIGN(16,100000 * sizeof(uint)));
    uint i = h % 4096;

    if (slot[i] == 0)
        return NULL;
    return (khi *)((char *)k + slot[i]);
}

__device__ __forceinline__ static khi* KERN_HASH_NEXT_ITEM(kds *k, khi *i)
{
    if (!i || i->next == 0)
        return NULL;
    return (khi *)((char *)k + i->next);
}

__global__ void gpu_hash_join(dti *dlhs, kds *drhs)
{
    uint i = 0,hash = 0,cid = 0;
    __shared__ uint crc32t[256];
    for (i = get_local_id(); i < 256; i += get_local_size())
    {
        crc32t[i] = drhs->crc32t[i];
    }
    __syncthreads();

    cid = dlhs[get_global_id()].cid;
    INIT_LEGACY_CRC32(hash);
    COMP_CRC32_REFLECTED_TABLE(hash, &cid, sizeof(int), crc32t);
    FIN_LEGACY_CRC32(hash);

    khi *item = KERN_HASH_FIRST_ITEM(drhs, hash);
    do
    {
        if (item && (item->hash == hash) && (cid == item->t.nid))
        {
            dlhs[get_global_id()].mat = 1;
        }
        item = KERN_HASH_NEXT_ITEM(drhs, item);
    } while (item != NULL);
}

int main(int argc, char *argv[])
{
    int devID;
    cudaDeviceProp deviceProps;

    printf("[%s] - Starting...\n", argv[0]);
    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);
    // get device name
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);

    kds *rhs = NULL;
    uint rsize = TYPEALIGN(16,sizeof(kds)) + (100000 + 4096) * sizeof(uint) + 100000 * sizeof(khi);
    checkCudaErrors(cudaMallocHost((void **)&rhs, rsize));
    memset(rhs, 0, sizeof(kds));
    for (uint i = 0;i < 256;i++)
    {
        rhs->crc32t[i] = crc32df[i];
    }

    rhs->length = rsize;
    rhs->ncols = 4;
    rhs->nitems = 100000;
    rhs->nslots = 4096;
    rhs->hash_min = 0;
    rhs->hash_max = UINT_MAX;

    uint *row_index = (uint *)(((char *)(rhs) + KERN_DATA_STORE_HEAD_LENGTH(kds)));
    uint *hash_slot = (uint *)(((char *)(rhs) + KERN_DATA_STORE_HEAD_LENGTH(kds)) + TYPEALIGN(16,100000 * sizeof(uint)));
    memset(row_index, 0, (100000 + 4096) * sizeof(uint));

    khi *row;
    uint i,j,k;
    for (i = 0;i < 100000;i++)
    {
        row = (khi *)((char *)rhs + TYPEALIGN(16,sizeof(kds)) + (100000 + 4096) * sizeof(uint) + i * sizeof(khi));
        row->hash = HOST_HASH_VALUE(i);
        row->rowid = i;
        row->t.nid = i;
        row->t.xid = (float)(i+1);
        row->t.yid = (float)(i+2);
        row->t.zid = (float)(i+3);
        row_index[row->rowid] = (uint)((uintptr_t)&row->t - (uintptr_t)rhs);
        j = row->hash % 4096;
        row->next = hash_slot[j];
        hash_slot[j] = (uintptr_t)row - (uintptr_t)rhs;
    }

    int n = 128 * 1024 * 1024;
    int nbytes = n * sizeof(dti);
    // allocate host memory
    dti *lhs = NULL;
    checkCudaErrors(cudaMallocHost((void **)&lhs, nbytes));
    for (k = 0;k < n;k++)
    {
        lhs[k].cid = k;
        lhs[k].mat = 0;
    }

    // allocate device memory
    dti *dlhs = NULL;
    checkCudaErrors(cudaMalloc((void **)&dlhs, nbytes));
    checkCudaErrors(cudaMemset(dlhs, 0, nbytes));
    kds *drhs = NULL;
    checkCudaErrors(cudaMalloc((void **)&drhs, rsize));
    checkCudaErrors(cudaMemset(drhs, 0, rsize));

    // set kernel launch configuration
    dim3 threads = dim3(1024, 1);
    dim3 blocks  = dim3(n / threads.x, 1);

    // create cuda event handles
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;
    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(drhs, rhs, rsize, cudaMemcpyHostToDevice, 0);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpyAsync(dlhs, lhs, nbytes, cudaMemcpyHostToDevice, 0);
    gpu_hash_join<<<blocks, threads, 0, 0>>>(dlhs, drhs);
    cudaMemcpyAsync(lhs, dlhs, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter=0;

    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    int cnt = 0;
    for (uint i = 0;i < n;i++)
    {
        if (lhs[i].mat == 1)
            cnt += 1;
    }
    printf("all match item num: %d\n", cnt);

    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(lhs));
    checkCudaErrors(cudaFreeHost(rhs));
    checkCudaErrors(cudaFree(dlhs));
    checkCudaErrors(cudaFree(drhs));

    return EXIT_SUCCESS;
}
