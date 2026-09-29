#include "ugds_internal.h"

#include <cstring>
#include <cstdio>
#include <cerrno>
#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <poll.h>
#include <unistd.h>
#include <time.h>

static bool ugds_profile_io_enabled()
{
    static int enabled = []() {
        const char* v = getenv("UGDS_PROFILE_IO");
        return (v != nullptr && v[0] != '\0' && strcmp(v, "0") != 0) ? 1 : 0;
    }();
    return enabled != 0;
}

static uint64_t ugds_profile_now_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int sync_prp_pool_alloc(PRPPool* pool)
{
    for (int word = 0; word < UGDS_PRP_POOL_BITMAP_WORDS; ++word) {
        if (pool->free_bitmap[word] == 0) continue;
        int bit = __builtin_ctzll(pool->free_bitmap[word]);
        int idx = word * 64 + bit;
        if (static_cast<size_t>(idx) >= pool->n_pages)
            return -1;
        pool->free_bitmap[word] &= ~(1ULL << bit);
        return idx;
    }
    return -1;
}

static void sync_prp_pool_free(PRPPool* pool, int idx)
{
    pool->free_bitmap[idx / 64] |= (1ULL << (idx % 64));
}

static nvm_cpl_t* wait_for_completion(HandleState* hs, IOQueuePair& qp)
{
    if (qp.irq_efd < 0) {
        nvm_cpl_t* cpl = nullptr;
        uint64_t spins = 0;
        const uint64_t max_spins = (uint64_t)hs->ctrl->timeout * 1000000ULL;
        while ((cpl = nvm_cq_dequeue(&qp.cq)) == nullptr) {
            if (++spins > max_spins) break;
            __builtin_ia32_pause();
        }
        return cpl;
    }

    // Absolute deadline prevents EINTR restarts from stretching total wait.
    const long timeout_ms = (long)hs->ctrl->timeout;
    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += timeout_ms / 1000;
    deadline.tv_nsec += (timeout_ms % 1000) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }

    for (;;) {
        // Poll CQ before blocking to prevent lost wakeups.
        nvm_cpl_t* cpl = nvm_cq_dequeue(&qp.cq);
        if (cpl != nullptr) return cpl;

        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        struct timespec ts;
        ts.tv_sec = deadline.tv_sec - now.tv_sec;
        ts.tv_nsec = deadline.tv_nsec - now.tv_nsec;
        if (ts.tv_nsec < 0) {
            ts.tv_sec -= 1;
            ts.tv_nsec += 1000000000L;
        }
        if (ts.tv_sec < 0)
            return nvm_cq_dequeue(&qp.cq);

        struct pollfd pfd = {qp.irq_efd, POLLIN, 0};
        int pr = ppoll(&pfd, 1, &ts, nullptr);
        if (pr < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "uGDS: ppoll(eventfd=%d) failed: %s\n",
                    qp.irq_efd, strerror(errno));
            return nullptr;
        }
        if (pr == 0)
            return nvm_cq_dequeue(&qp.cq);

        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            fprintf(stderr, "uGDS: eventfd=%d invalid, revents=0x%x\n",
                    qp.irq_efd, pfd.revents);
            return nullptr;
        }
        if (!(pfd.revents & POLLIN))
            continue;

        uint64_t cnt;
        ssize_t r = read(qp.irq_efd, &cnt, sizeof(cnt));
        if (r == (ssize_t)sizeof(cnt))
            continue;
        if (r < 0 && (errno == EINTR || errno == EAGAIN))
            continue;
        fprintf(stderr, "uGDS: read(eventfd=%d) failed: %s\n",
                qp.irq_efd, r < 0 ? strerror(errno) : "short read");
        return nullptr;
    }
}

static int drain_sync_completion(HandleState* hs, IOQueuePair& qp,
                                 size_t* bytes_done, uint16_t* in_flight,
                                 uint64_t* profile_wait_ns, bool profile_io)
{
    uint64_t profile_wait_start_ns = profile_io ? ugds_profile_now_ns() : 0;
    nvm_cpl_t* cpl = wait_for_completion(hs, qp);
    if (profile_io) *profile_wait_ns += ugds_profile_now_ns() - profile_wait_start_ns;
    if (cpl == nullptr) {
        return -ETIMEDOUT;
    }

    uint16_t cid = *NVM_CPL_CID(cpl);
    uint16_t status = UGDS_CPL_SCT_SC(cpl);

    nvm_sq_update(&qp.sq);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    nvm_cq_update(&qp.cq);

    if (cid >= qp.sync_cmd_map.size() || !qp.sync_cmd_map[cid].active) {
        fprintf(stderr, "uGDS: unexpected sync completion cid=%u\n", cid);
        return -EIO;
    }

    CmdSlot& slot = qp.sync_cmd_map[cid];
    if (slot.prp_page_idx != UINT16_MAX) {
        sync_prp_pool_free(&qp.sync_prp_pool, slot.prp_page_idx);
    }
    slot.active = false;
    (*in_flight)--;

    if (status != 0) {
        return -EIO;
    }

    *bytes_done += slot.chunk_bytes;
    return 0;
}

ssize_t do_io_internal(uGDSHandle_t fh, void* bufPtr_base, size_t size,
                       off_t file_offset, off_t bufPtr_offset, uint8_t opcode)
{
    HandleState* hs = static_cast<HandleState*>(fh);

    if (hs == nullptr || bufPtr_base == nullptr || size == 0) {
        return -EINVAL;
    }
    if (file_offset < 0 || bufPtr_offset < 0) {
        return -EINVAL;
    }
    if (hs->block_size == 0) {
        return -EINVAL;
    }
    if ((static_cast<size_t>(file_offset) % hs->block_size) != 0 ||
        (size % hs->block_size) != 0) {
        return -EINVAL;
    }

    const size_t page_size = hs->ctrl->page_size;
    if (page_size == 0) {
        return -EINVAL;
    }

    nvm_dma_t* buf_dma = nullptr;
    bool on_the_fly = false;
    size_t buf_page_start = 0;

    {
        std::lock_guard<std::mutex> drv_lock(g_driver.lock);
        auto it = g_driver.buf_registry.find(bufPtr_base);
        if (it != g_driver.buf_registry.end()) {
            buf_dma = it->second.dma;
            it->second.in_flight.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    if (buf_dma != nullptr) {
        if ((static_cast<size_t>(bufPtr_offset) % page_size) != 0) {
            std::lock_guard<std::mutex> drv_lock(g_driver.lock);
            auto it = g_driver.buf_registry.find(bufPtr_base);
            if (it != g_driver.buf_registry.end())
                it->second.in_flight.fetch_sub(1, std::memory_order_acq_rel);
            return -EINVAL;
        }
        buf_page_start = static_cast<size_t>(bufPtr_offset) / page_size;
        /* Overflow-safe bounds check: compute pages_needed separately,
         * then verify buf_page_start + pages_needed <= n_ioaddrs
         * without risking wrap. */
        size_t pages_needed = (size - 1) / page_size + 1;
        if (pages_needed > buf_dma->n_ioaddrs ||
            buf_page_start > buf_dma->n_ioaddrs - pages_needed) {
            std::lock_guard<std::mutex> drv_lock(g_driver.lock);
            auto it = g_driver.buf_registry.find(bufPtr_base);
            if (it != g_driver.buf_registry.end())
                it->second.in_flight.fetch_sub(1, std::memory_order_acq_rel);
            return -EINVAL;
        }
    } else {
        void* map_ptr = static_cast<uint8_t*>(bufPtr_base) + bufPtr_offset;
        // Reject non-64KB-aligned on-the-fly buffers (kernel driver silently rounds down)
        if ((reinterpret_cast<uintptr_t>(map_ptr) & ((1UL << 16) - 1)) != 0) {
            return -EINVAL;
        }
        int rc = nvm_dma_map_device(&buf_dma, hs->ctrl, map_ptr, size);
        if (rc != 0 || buf_dma == nullptr) {
            return -EIO;
        }
        on_the_fly = true;
        buf_page_start = 0;
    }

    const uint64_t start_lba = static_cast<uint64_t>(file_offset) / hs->block_size;

    const size_t prp_capacity = page_size / sizeof(uint64_t);

    size_t max_xfer = hs->max_transfer_size;
    if (max_xfer == 0) {
        max_xfer = UGDS_DEFAULT_MAX_TRANSFER_SIZE;
    }
    if (max_xfer < page_size) {
        max_xfer = page_size;
    }
    // Cap by PRP list capacity: PRP1 + prp_capacity entries → (prp_capacity + 1) pages
    size_t prp_max = (prp_capacity + 1) * page_size;
    if (max_xfer > prp_max) {
        max_xfer = prp_max;
    }

    const uint16_t qp_idx = static_cast<uint16_t>(
        hs->rr_counter.fetch_add(1) % hs->num_qps);
    IOQueuePair& qp = *hs->qps[qp_idx];

    ssize_t result = 0;
    bool timed_out = false;
    size_t bytes_done = 0;
    uint64_t current_lba = start_lba;
    size_t current_page = buf_page_start;
    const bool profile_io = ugds_profile_io_enabled();
    const uint64_t profile_start_ns = profile_io ? ugds_profile_now_ns() : 0;
    uint64_t profile_submit_ns = 0;
    uint64_t profile_wait_ns = 0;
    uint64_t profile_commands = 0;
    uint16_t profile_window_depth = 1;
    uint16_t profile_max_in_flight = 1;

    {
        std::lock_guard<std::mutex> qp_lock(qp.lock);
        uint16_t window_depth = 1;
        uint16_t max_in_flight = 1;

        /* Re-check wedged after acquiring QP lock: a previous
         * operation may have timed out while we waited. */
        if (hs->wedged.load(std::memory_order_acquire)) {
            result = -EBADF;
            goto out;
        }

        if (hs->sync_window_enabled && qp.sync_prp_pool.dma != nullptr &&
            qp.sync_prp_pool.n_pages > 0 && !qp.sync_cmd_map.empty()) {
            window_depth = std::min<uint16_t>(
                hs->sync_window_depth,
                static_cast<uint16_t>(qp.sq.qs - 1));
            window_depth = std::min<uint16_t>(
                window_depth,
                static_cast<uint16_t>(qp.sync_prp_pool.n_pages));
            if (window_depth < 2)
                window_depth = 1;
        }
        profile_window_depth = window_depth;

        if (window_depth == 1) {
            while (bytes_done < size) {
                size_t remaining = size - bytes_done;
                size_t chunk_size = std::min(remaining, max_xfer);

                chunk_size = (chunk_size / hs->block_size) * hs->block_size;
                if (chunk_size == 0) {
                    result = -EINVAL;
                    goto out;
                }

                size_t n_blocks = chunk_size / hs->block_size;
                size_t n_pages = (chunk_size + page_size - 1) / page_size;
                if (n_pages == 0) n_pages = 1;

                nvm_cmd_t* cmd = nullptr;
                while ((cmd = nvm_sq_enqueue(&qp.sq)) == nullptr) {
                    nvm_cpl_t* drain = wait_for_completion(hs, qp);
                    if (drain == nullptr) {
                        result = -EIO;
                        /* An older submitted command may still DMA. Treat this
                         * exactly like a post-submit completion timeout. */
                        timed_out = true;
                        hs->wedged.store(true, std::memory_order_release);
                        goto out;
                    }
                    uint16_t st = UGDS_CPL_SCT_SC(drain);
                    nvm_sq_update(&qp.sq);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    nvm_cq_update(&qp.cq);
                    if (st != 0) {
                        result = -EIO;
                        goto out;
                    }
                }

                memset(cmd, 0, sizeof(nvm_cmd_t));
                uint16_t cid = NVM_DEFAULT_CID(&qp.sq);
                nvm_cmd_header(cmd, cid, opcode, hs->ns_id);

                if (n_pages == 1) {
                    nvm_cmd_data_ptr(cmd, buf_dma->ioaddrs[current_page], 0);
                } else if (n_pages == 2) {
                    nvm_cmd_data_ptr(cmd,
                                     buf_dma->ioaddrs[current_page],
                                     buf_dma->ioaddrs[current_page + 1]);
                } else {
                    volatile uint64_t* prp_list =
                        reinterpret_cast<volatile uint64_t*>(qp.prp_dma->vaddr);
                    for (size_t i = 1; i < n_pages; ++i) {
                        prp_list[i - 1] = buf_dma->ioaddrs[current_page + i];
                    }
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    nvm_cmd_data_ptr(cmd,
                                     buf_dma->ioaddrs[current_page],
                                     qp.prp_dma->ioaddrs[0]);
                }

                nvm_cmd_rw_blks(cmd, current_lba, static_cast<uint16_t>(n_blocks));

                std::atomic_thread_fence(std::memory_order_seq_cst);
                uint64_t profile_submit_start_ns = profile_io ? ugds_profile_now_ns() : 0;
                nvm_sq_submit(&qp.sq);
                if (profile_io) profile_submit_ns += ugds_profile_now_ns() - profile_submit_start_ns;
                std::atomic_thread_fence(std::memory_order_seq_cst);

                uint64_t profile_wait_start_ns = profile_io ? ugds_profile_now_ns() : 0;
                nvm_cpl_t* cpl = wait_for_completion(hs, qp);
                if (profile_io) profile_wait_ns += ugds_profile_now_ns() - profile_wait_start_ns;
                if (profile_io) ++profile_commands;
                if (cpl == nullptr) {
                    result = -EIO;
                    timed_out = true;
                    /* Mark wedged while still holding qp.lock to prevent
                     * QP reuse before the flag is visible. */
                    hs->wedged.store(true, std::memory_order_release);
                    goto out;
                }

                uint16_t status = UGDS_CPL_SCT_SC(cpl);

                nvm_sq_update(&qp.sq);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                nvm_cq_update(&qp.cq);

                if (status != 0) {
                    result = -EIO;
                    goto out;
                }

                bytes_done += chunk_size;
                current_lba += n_blocks;
                current_page += n_pages;
            }
        } else {
            size_t submit_bytes = 0;
            size_t submit_page = current_page;
            uint64_t submit_lba = current_lba;
            uint16_t in_flight = 0;

            for (CmdSlot& slot : qp.sync_cmd_map)
                slot.active = false;

            while (bytes_done < size) {
                bool submitted = false;

                while (result >= 0 && submit_bytes < size &&
                       in_flight < window_depth) {
                    size_t remaining = size - submit_bytes;
                    size_t chunk_size = std::min(remaining, max_xfer);

                    chunk_size = (chunk_size / hs->block_size) * hs->block_size;
                    if (chunk_size == 0) {
                        result = -EINVAL;
                        break;
                    }

                    size_t n_blocks = chunk_size / hs->block_size;
                    size_t n_pages = (chunk_size + page_size - 1) / page_size;
                    if (n_pages == 0) n_pages = 1;

                    uint16_t prp_idx = UINT16_MAX;
                    if (n_pages > 2) {
                        int pidx = sync_prp_pool_alloc(&qp.sync_prp_pool);
                        if (pidx < 0) {
                            break;
                        }
                        prp_idx = static_cast<uint16_t>(pidx);
                    }

                    uint16_t slot = static_cast<uint16_t>(
                        qp.sq.tail.load(std::memory_order_relaxed) % qp.sq.qs);
                    if (slot >= qp.sync_cmd_map.size() ||
                        qp.sync_cmd_map[slot].active) {
                        if (prp_idx != UINT16_MAX)
                            sync_prp_pool_free(&qp.sync_prp_pool, prp_idx);
                        result = -EIO;
                        timed_out = true;
                        hs->wedged.store(true, std::memory_order_release);
                        break;
                    }

                    nvm_cmd_t* cmd = nvm_sq_enqueue(&qp.sq);
                    if (cmd == nullptr) {
                        if (prp_idx != UINT16_MAX)
                            sync_prp_pool_free(&qp.sync_prp_pool, prp_idx);
                        break;
                    }

                    memset(cmd, 0, sizeof(nvm_cmd_t));
                    nvm_cmd_header(cmd, slot, opcode, hs->ns_id);

                    if (n_pages == 1) {
                        nvm_cmd_data_ptr(cmd, buf_dma->ioaddrs[submit_page], 0);
                    } else if (n_pages == 2) {
                        nvm_cmd_data_ptr(cmd,
                                         buf_dma->ioaddrs[submit_page],
                                         buf_dma->ioaddrs[submit_page + 1]);
                    } else {
                        volatile uint64_t* prp_list =
                            reinterpret_cast<volatile uint64_t*>(
                                static_cast<uint8_t*>(qp.sync_prp_pool.buf) +
                                prp_idx * page_size);
                        for (size_t i = 1; i < n_pages; ++i) {
                            prp_list[i - 1] = buf_dma->ioaddrs[submit_page + i];
                        }
                        std::atomic_thread_fence(std::memory_order_seq_cst);
                        nvm_cmd_data_ptr(cmd,
                                         buf_dma->ioaddrs[submit_page],
                                         qp.sync_prp_pool.dma->ioaddrs[prp_idx]);
                    }

                    nvm_cmd_rw_blks(cmd, submit_lba,
                                    static_cast<uint16_t>(n_blocks));

                    CmdSlot& cs = qp.sync_cmd_map[slot];
                    cs.io_idx = 0;
                    cs.chunk_bytes = chunk_size;
                    cs.prp_page_idx = prp_idx;
                    cs.active = true;

                    submit_bytes += chunk_size;
                    submit_lba += n_blocks;
                    submit_page += n_pages;
                    ++in_flight;
                    if (in_flight > max_in_flight)
                        max_in_flight = in_flight;
                    ++profile_commands;
                    submitted = true;
                }

                if (submitted) {
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    uint64_t profile_submit_start_ns =
                        profile_io ? ugds_profile_now_ns() : 0;
                    nvm_sq_submit(&qp.sq);
                    if (profile_io)
                        profile_submit_ns +=
                            ugds_profile_now_ns() - profile_submit_start_ns;
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                }

                if (!submitted && in_flight == 0 && submit_bytes < size &&
                    result >= 0) {
                    result = -EIO;
                    goto out;
                }

                if (result < 0) {
                    while (in_flight > 0) {
                        int rc = drain_sync_completion(hs, qp, &bytes_done,
                                                       &in_flight,
                                                       &profile_wait_ns,
                                                       profile_io);
                        if (rc == -ETIMEDOUT) {
                            timed_out = true;
                            hs->wedged.store(true, std::memory_order_release);
                            break;
                        }
                    }
                    goto out;
                }

                if (in_flight == 0)
                    continue;

                int rc = drain_sync_completion(hs, qp, &bytes_done, &in_flight,
                                               &profile_wait_ns, profile_io);
                if (rc == -ETIMEDOUT) {
                    result = -EIO;
                    timed_out = true;
                    hs->wedged.store(true, std::memory_order_release);
                    goto out;
                }
                if (rc != 0) {
                    result = -EIO;
                }
            }

            profile_max_in_flight = max_in_flight;
        }

        result = static_cast<ssize_t>(bytes_done);

        /* Transfer timeout resources to the QP while we still hold
         * qp.lock.  A concurrent force teardown also acquires qp.lock
         * in cleanup_timeout_resources(), so this prevents the race
         * where teardown destroys the QP before we store the mapping. */
        if (timed_out) {
            if (on_the_fly) {
                qp.timeout_dma = buf_dma;
            } else {
                qp.timeout_registered_buf = bufPtr_base;
            }
            buf_dma = nullptr;
        }

    out:;
    }

    if (timed_out) {
        /* NVMe command may still be executing. wedged was set
         * under qp.lock. Resources were transferred to the QP above. */
        fprintf(stderr, "uGDS: I/O timeout -- handle wedged. "
                "Controller reset required.\n");
    } else if (on_the_fly && buf_dma != nullptr) {
        nvm_dma_unmap(buf_dma);
    } else if (!on_the_fly && buf_dma != nullptr) {
        /* Registered buffer: release in-flight reference. */
        std::lock_guard<std::mutex> drv_lock(g_driver.lock);
        auto it = g_driver.buf_registry.find(bufPtr_base);
        if (it != g_driver.buf_registry.end())
            it->second.in_flight.fetch_sub(1, std::memory_order_acq_rel);
    }

    if (profile_io) {
        const uint64_t total_ns = ugds_profile_now_ns() - profile_start_ns;
        fprintf(stderr,
                "UGDS_PROFILE_IO op=%s size=%zu file_offset=%lld buf_offset=%lld "
                "max_xfer=%zu commands=%llu qp=%u result=%zd total_us=%.3f "
                "submit_us=%.3f wait_us=%.3f on_the_fly=%d interrupt=%d "
                "window_depth=%u max_in_flight=%u\n",
                opcode == NVM_IO_READ ? "read" : "write",
                size,
                (long long)file_offset,
                (long long)bufPtr_offset,
                max_xfer,
                (unsigned long long)profile_commands,
                (unsigned)qp_idx,
                result,
                (double)total_ns / 1000.0,
                (double)profile_submit_ns / 1000.0,
                (double)profile_wait_ns / 1000.0,
                on_the_fly ? 1 : 0,
                hs->interrupt_mode ? 1 : 0,
                (unsigned)profile_window_depth,
                (unsigned)profile_max_in_flight);
    }

    return result;
}

extern "C" ssize_t uGDSRead(uGDSHandle_t fh, void* bufPtr_base, size_t size,
                              off_t file_offset, off_t bufPtr_offset)
{
    if (fh == nullptr) return -EINVAL;
    std::shared_ptr<HandleState> hs_sp;
    HandleState* hs = handle_lookup(fh, &hs_sp);
    if (!hs) return -EBADF;
    ssize_t ret = do_io_internal(fh, bufPtr_base, size, file_offset, bufPtr_offset, NVM_IO_READ);
    handle_release(hs);
    return ret;
}

extern "C" ssize_t uGDSWrite(uGDSHandle_t fh, const void* bufPtr_base, size_t size,
                               off_t file_offset, off_t bufPtr_offset)
{
    if (fh == nullptr) return -EINVAL;
    std::shared_ptr<HandleState> hs_sp;
    HandleState* hs = handle_lookup(fh, &hs_sp);
    if (!hs) return -EBADF;
    ssize_t ret = do_io_internal(fh, const_cast<void*>(bufPtr_base), size, file_offset, bufPtr_offset,
                 NVM_IO_WRITE);
    handle_release(hs);
    return ret;
}
