#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    // Exercise enough commands to wrap the shallow sync SQ/CQ many times.
    setenv("UGDS_SYNC_IO_WINDOW_DEPTH", "16", 1);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const size_t io_size = 16 * 1024 * 1024;
    const unsigned rounds = 128;
    const uint32_t pattern = 0x51AFC0DE;
    const size_t n_words = io_size / sizeof(uint32_t);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, io_size);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, io_size, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    fill_pattern_u32<<<(n_words + 255) / 256, 256>>>(
        (uint32_t*)d_buf, pattern, n_words);
    cudaDeviceSynchronize();

    for (unsigned i = 0; i < rounds; ++i) {
        ssize_t ret = uGDSWrite(fh, d_buf, io_size, 0, 0);
        if (ret != (ssize_t)io_size)
            TEST_FAIL("round %u uGDSWrite: %zd / %zu", i, ret, io_size);
    }

    for (unsigned i = 0; i < rounds; ++i) {
        cudaMemset(d_buf, 0, io_size);
        cudaDeviceSynchronize();
        ssize_t ret = uGDSRead(fh, d_buf, io_size, 0, 0);
        if (ret != (ssize_t)io_size)
            TEST_FAIL("round %u uGDSRead: %zd / %zu", i, ret, io_size);
    }

    uint32_t* h_buf = (uint32_t*)malloc(io_size);
    cudaMemcpy(h_buf, d_buf, io_size, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n_words; ++i) {
        if (h_buf[i] != pattern) {
            free(h_buf);
            TEST_FAIL("mismatch at word %zu: 0x%08X != 0x%08X",
                      i, h_buf[i], pattern);
        }
    }
    free(h_buf);

    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
