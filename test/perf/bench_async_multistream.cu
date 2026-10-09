#include <ugds.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#include "bench_utils.h"

static uint64_t elapsed_ns(const timespec& start, const timespec& end) {
    return static_cast<uint64_t>(end.tv_sec - start.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(end.tv_nsec - start.tv_nsec);
}

static void cuda_check(cudaError_t rc, const char* operation) {
    if (rc != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(rc));
        exit(EXIT_FAILURE);
    }
}

static void ugds_check(uGDSError_t status, const char* operation) {
    if (status.err != UGDS_SUCCESS) {
        fprintf(stderr, "%s failed: %s\n", operation,
                uGDS_status_error(status.err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr,
                "Usage: %s <device> <gpu_id> <length> <io_size> <streams>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char* device = argv[1];
    int gpu_id = atoi(argv[2]);
    size_t length = parse_size(argv[3]);
    size_t io_size = parse_size(argv[4]);
    unsigned n_streams = static_cast<unsigned>(strtoul(argv[5], nullptr, 10));
    if (length == 0 || io_size == 0 || n_streams == 0 ||
        length % io_size != 0 || io_size % 4096 != 0) {
        fprintf(stderr, "invalid length, io_size, or stream count\n");
        return EXIT_FAILURE;
    }

    cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice");
    ugds_check(uGDSDriverOpen(), "uGDSDriverOpen");

    int fd = open(device, O_RDWR);
    if (fd < 0) {
        perror("open");
        return EXIT_FAILURE;
    }

    uGDSDescr_t descriptor = {};
    descriptor.type = UGDS_HANDLE_TYPE_OPAQUE_FD;
    descriptor.handle.fd = fd;
    uGDSHandle_t handle = nullptr;
    ugds_check(uGDSHandleRegister(&handle, &descriptor), "uGDSHandleRegister");

    const size_t buffer_size = io_size * n_streams;
    void* gpu_buffer = nullptr;
    cuda_check(cudaMalloc(&gpu_buffer, buffer_size), "cudaMalloc");
    cuda_check(cudaMemset(gpu_buffer, 0, buffer_size), "cudaMemset");
    ugds_check(uGDSBufRegister(gpu_buffer, buffer_size, 0), "uGDSBufRegister");

    std::vector<cudaStream_t> streams(n_streams);
    for (unsigned i = 0; i < n_streams; ++i) {
        cuda_check(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
        ugds_check(uGDSStreamRegister(streams[i]), "uGDSStreamRegister");
    }

    std::vector<size_t> sizes(n_streams, io_size);
    std::vector<off_t> file_offsets(n_streams);
    std::vector<off_t> buffer_offsets(n_streams);
    std::vector<ssize_t> completed(n_streams, 0);

    size_t done = 0;
    uint64_t operations = 0;
    timespec start = {}, end = {};
    clock_gettime(CLOCK_MONOTONIC, &start);

    while (done < length) {
        unsigned submitted = 0;
        for (; submitted < n_streams && done + submitted * io_size < length;
             ++submitted) {
            sizes[submitted] = io_size;
            file_offsets[submitted] =
                static_cast<off_t>(done + submitted * io_size);
            buffer_offsets[submitted] =
                static_cast<off_t>(submitted * io_size);
            completed[submitted] = 0;
            ugds_check(
                uGDSReadAsync(handle, gpu_buffer, &sizes[submitted],
                              &file_offsets[submitted], &buffer_offsets[submitted],
                              &completed[submitted], streams[submitted]),
                "uGDSReadAsync");
        }

        for (unsigned i = 0; i < submitted; ++i)
            cuda_check(cudaStreamSynchronize(streams[i]), "cudaStreamSynchronize");
        for (unsigned i = 0; i < submitted; ++i) {
            if (completed[i] != static_cast<ssize_t>(io_size)) {
                fprintf(stderr, "short completion[%u]: %zd / %zu\n",
                        i, completed[i], io_size);
                return EXIT_FAILURE;
            }
        }

        done += static_cast<size_t>(submitted) * io_size;
        operations += submitted;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double seconds = elapsed_ns(start, end) / 1e9;
    double bandwidth_mib_s = (done / static_cast<double>(MB)) / seconds;
    printf("{\n");
    printf("  \"label\": \"uGDS-async-multistream-read\",\n");
    printf("  \"io_size\": %zu,\n", io_size);
    printf("  \"streams\": %u,\n", n_streams);
    printf("  \"total_ops\": %llu,\n",
           static_cast<unsigned long long>(operations));
    printf("  \"bandwidth_mbps\": %.2f\n", bandwidth_mib_s);
    printf("}\n");

    for (auto stream : streams) {
        ugds_check(uGDSStreamDeregister(stream), "uGDSStreamDeregister");
        cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    }
    ugds_check(uGDSBufDeregister(gpu_buffer), "uGDSBufDeregister");
    cuda_check(cudaFree(gpu_buffer), "cudaFree");
    uGDSHandleDeregister(handle);
    close(fd);
    ugds_check(uGDSDriverClose(), "uGDSDriverClose");
    return EXIT_SUCCESS;
}
