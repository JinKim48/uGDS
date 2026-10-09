#include "test_utils.h"

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;
    cudaSetDevice(g_gpu_id);

    uGDSError_t st = uGDSDriverOpen();
    ASSERT_OK(st, "DriverOpen");

    uGDSHandle_t fh = open_handle();
    if (!fh) TEST_FAIL("open_handle failed");

    const unsigned n_entries = 4;
    const size_t io_size = 16UL * 1024 * 1024;
    const size_t total = n_entries * io_size;
    const size_t words_per_entry = io_size / sizeof(uint32_t);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, total);
    if (!d_buf) TEST_FAIL("cudaMalloc failed");

    st = uGDSBufRegister(d_buf, total, TEST_BUF_FLAGS);
    ASSERT_OK(st, "BufRegister");

    for (unsigned i = 0; i < n_entries; ++i) {
        uint32_t pattern = 0xB4000000U | i;
        auto* ptr = reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(d_buf) + i * io_size);
        fill_pattern_u32<<<(words_per_entry + 255) / 256, 256>>>(
            ptr, pattern, words_per_entry);
    }
    cudaDeviceSynchronize();

    for (unsigned i = 0; i < n_entries; ++i) {
        ssize_t ret = uGDSWrite(fh, d_buf, io_size,
                                static_cast<off_t>(i * io_size),
                                static_cast<off_t>(i * io_size));
        if (ret != static_cast<ssize_t>(io_size))
            TEST_FAIL("sync write[%u] returned %zd", i, ret);
    }

    cudaMemset(d_buf, 0, total);
    cudaDeviceSynchronize();

    uGDSBatchHandle_t batch = nullptr;
    st = uGDSBatchIOSetUp(&batch, fh, n_entries);
    ASSERT_OK(st, "BatchIOSetUp");

    uGDSIOParams_t params[n_entries];
    for (unsigned i = 0; i < n_entries; ++i) {
        memset(&params[i], 0, sizeof(params[i]));
        params[i].devPtr_base = d_buf;
        params[i].file_offset = static_cast<off_t>(i * io_size);
        params[i].devPtr_offset = static_cast<off_t>(i * io_size);
        params[i].size = io_size;
        params[i].opcode = UGDS_READ;
        params[i].cookie = reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1));
    }

    st = uGDSBatchIOSubmit(batch, n_entries, params, 0);
    ASSERT_OK(st, "BatchIOSubmit");

    uGDSIOEvents_t events[n_entries];
    unsigned nr_out = n_entries;
    struct timespec timeout = {30, 0};
    st = uGDSBatchIOGetStatus(batch, n_entries, &nr_out, events, &timeout);
    ASSERT_OK(st, "BatchIOGetStatus");
    if (nr_out != n_entries)
        TEST_FAIL("expected %u events, got %u", n_entries, nr_out);

    for (unsigned i = 0; i < nr_out; ++i) {
        if (events[i].status != UGDS_BATCH_COMPLETE)
            TEST_FAIL("event[%u] status 0x%x", i, events[i].status);
        if (events[i].ret != static_cast<ssize_t>(io_size))
            TEST_FAIL("event[%u] ret %zd", i, events[i].ret);
    }

    auto* h_buf = static_cast<uint32_t*>(malloc(total));
    if (!h_buf) TEST_FAIL("host allocation failed");
    cudaMemcpy(h_buf, d_buf, total, cudaMemcpyDeviceToHost);
    for (unsigned i = 0; i < n_entries; ++i) {
        uint32_t expected = 0xB4000000U | i;
        size_t base = i * words_per_entry;
        for (size_t word = 0; word < words_per_entry; ++word) {
            if (h_buf[base + word] != expected) {
                uint32_t actual = h_buf[base + word];
                free(h_buf);
                TEST_FAIL("entry[%u] word %zu: 0x%08X != 0x%08X",
                          i, word, actual, expected);
            }
        }
    }
    free(h_buf);

    uGDSBatchIODestroy(batch);
    uGDSBufDeregister(d_buf);
    cudaFree(d_buf);
    close_handle(fh);
    uGDSDriverClose();
    TEST_PASS();
}
