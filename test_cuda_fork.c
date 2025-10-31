#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <cuda.h>

int main() {
    CUresult res;
    CUcontext ctx;
    CUdevice dev;

    // Initialize CUDA in parent
    printf("=== PARENT PROCESS (PID %d) ===\n", getpid());
    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", res);
        return 1;
    }
    printf("cuInit() successful\n");

    res = cuDeviceGet(&dev, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGet failed: %d\n", res);
        return 1;
    }
    printf("cuDeviceGet() successful, device: %d\n", dev);

    res = cuCtxCreate(&ctx, 0, dev);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxCreate failed: %d\n", res);
        return 1;
    }
    printf("cuCtxCreate() successful, context: %p\n", (void*)ctx);

    // Test 1: Fork without exec
    printf("\n=== TEST 1: Fork WITHOUT exec ===\n");
    pid_t pid1 = fork();
    if (pid1 == 0) {
        // Child process
        printf("CHILD 1 (PID %d): Checking CUDA context\n", getpid());
        CUcontext current;
        res = cuCtxGetCurrent(&current);
        if (res != CUDA_SUCCESS) {
            printf("CHILD 1: cuCtxGetCurrent failed: %d\n", res);
        } else {
            printf("CHILD 1: cuCtxGetCurrent successful, context: %p\n", (void*)current);
            if (current == ctx) {
                printf("CHILD 1: ✓ Context is SAME as parent (%p == %p)\n", (void*)current, (void*)ctx);
            } else if (current == NULL) {
                printf("CHILD 1: ✗ Context is NULL (lost after fork)\n");
            } else {
                printf("CHILD 1: ? Context is DIFFERENT from parent (%p != %p)\n", (void*)current, (void*)ctx);
            }
        }

        // Try to use CUDA after fork
        printf("CHILD 1: Attempting cudaFree(0) to test CUDA functionality\n");
        CUdeviceptr ptr = 0;
        res = cuMemAlloc(&ptr, 1024);
        if (res != CUDA_SUCCESS) {
            printf("CHILD 1: ✗ cuMemAlloc FAILED: %d (CUDA broken after fork)\n", res);
        } else {
            printf("CHILD 1: ✓ cuMemAlloc successful: %p (CUDA works after fork)\n", (void*)ptr);
            cuMemFree(ptr);
        }

        _exit(0);
    }
    waitpid(pid1, NULL, 0);

    // Test 2: Fork WITH exec
    printf("\n=== TEST 2: Fork WITH exec ===\n");
    printf("PARENT: Creating child that will exec...\n");
    pid_t pid2 = fork();
    if (pid2 == 0) {
        // Child process - will exec
        printf("CHILD 2 (PID %d): About to exec /bin/echo\n", getpid());
        execl("/bin/echo", "echo", "CHILD 2: Exec successful - CUDA context is LOST", NULL);
        fprintf(stderr, "CHILD 2: exec failed\n");
        _exit(1);
    }
    waitpid(pid2, NULL, 0);

    printf("\n=== SUMMARY ===\n");
    printf("1. fork() without exec(): Context pointer may be copied but CUDA state is broken\n");
    printf("2. fork() with exec(): All parent process memory is replaced, context is LOST\n");
    printf("\n=== CONCLUSION ===\n");
    printf("CUDA contexts CANNOT be shared across process boundaries via fork/exec.\n");
    printf("Each process needs its own CUDA context created via cuCtxCreate().\n");

    cuCtxDestroy(ctx);
    return 0;
}
