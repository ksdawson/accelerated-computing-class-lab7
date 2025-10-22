#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

void rle_compress_cpu(
    uint32_t raw_count,
    char const *raw,
    std::vector<char> &compressed_data,
    std::vector<uint32_t> &compressed_lengths) {
    compressed_data.clear();
    compressed_lengths.clear();

    uint32_t i = 0;
    while (i < raw_count) {
        char c = raw[i];
        uint32_t run_length = 1;
        i++;
        while (i < raw_count && raw[i] == c) {
            run_length++;
            i++;
        }
        compressed_data.push_back(c);
        compressed_lengths.push_back(run_length);
    }
}

/// <--- your code here --->

struct __align__(4) RleData {
    char val;
    uint32_t count : 24;
};
static_assert(sizeof(RleData) == 4, "RleData must be 4 bytes");
struct __align__(8) RleData2 {
    uint32_t count;
    char val;
};
static_assert(sizeof(RleData2) == 8, "RleData2 must be 8 bytes");

__device__ RleData create_rle_data(char val, uint32_t count) {
    RleData data;
    data.val = val;
    data.count = count;
    return data;
}
__device__ RleData2 create_rle_data2(char val, uint32_t count) {
    RleData2 data;
    data.val = val;
    data.count = count;
    return data;
}

struct SumOp {
    using Data = RleData;

    static __host__ __device__ __forceinline__ Data identity() {
        Data data;
        data.val = '\0';
        data.count = 0;
        return data;
    }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        b.count += a.count;
        return b;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Scan code

/// Helpers to deal with Op::Data type
// Generic, aligned struct for vectorized memory access
template <typename T, int N>
struct alignas(sizeof(T) * N) Vectorized {
    T elements[N];
};
// Needed because compiler doesn't know how to shuffle DebugRange
template <typename T>
__device__ T shfl_up_any(T val, unsigned int delta) {
    T result;
    if constexpr (sizeof(T) == 4) {
        // Single 32-bit value
        uint32_t v = *reinterpret_cast<uint32_t*>(&val);
        v = __shfl_up_sync(0xffffffff, v, delta);
        *reinterpret_cast<uint32_t*>(&result) = v;
    } else {
        // Two 32-bit values (e.g. DebugRange)
        const uint32_t* src = reinterpret_cast<const uint32_t*>(&val);
        uint32_t* dst = reinterpret_cast<uint32_t*>(&result);
        dst[0] = __shfl_up_sync(0xffffffff, src[0], delta);
        dst[1] = __shfl_up_sync(0xffffffff, src[1], delta);
    }
    return result;
}
template <typename T>
__device__ T shfl_down_any(T val, unsigned int delta) {
    T result;
    if constexpr (sizeof(T) == 4) {
        // Single 32-bit value
        uint32_t v = *reinterpret_cast<uint32_t*>(&val);
        v = __shfl_down_sync(0xffffffff, v, delta);
        *reinterpret_cast<uint32_t*>(&result) = v;
    } else {
        // Two 32-bit values (e.g. DebugRange)
        const uint32_t* src = reinterpret_cast<const uint32_t*>(&val);
        uint32_t* dst = reinterpret_cast<uint32_t*>(&result);
        dst[0] = __shfl_down_sync(0xffffffff, src[0], delta);
        dst[1] = __shfl_down_sync(0xffffffff, src[1], delta);
    }
    return result;
}

namespace scan_gpu {

// Helpers
template <typename Op>
__device__ typename Op::Data warp_local_scan(typename Op::Data val) {
    using Data = typename Op::Data;

    // Computes parallel prefix on 32 elements using Hillis Steele Scan w/ warp shuffle
    const uint32_t thread_idx = threadIdx.x % 32;
    uint32_t idx = 1;
    for (uint32_t step = 0; step < 5; ++step) { // log2(32) = 5
        // Load prefix from register
        Data tmp = shfl_up_any(val, idx);
        tmp = (thread_idx >= idx) ? tmp : Op::identity(); // Mask out

        // Update prefix in register
        val = Op::combine(tmp, val);

        // Multiply idx by 2
        idx <<= 1;
    }

    return val;
}

template <typename Op, uint32_t VEC_SIZE, bool DO_FIX>
__device__ inline typename Op::Data thread_local_scan(size_t n, typename Op::Data const *x, typename Op::Data *out,
    const uint32_t start_i, const uint32_t end_i,
    typename Op::Data accumulator
) {
    using Data = typename Op::Data;
    using VecData = Vectorized<Data, VEC_SIZE>;

    // Vectorize
    VecData const *vx = reinterpret_cast<VecData const *>(x);
    VecData *vout = reinterpret_cast<VecData*>(out);
    const uint32_t start_vi = start_i / VEC_SIZE;
    const uint32_t end_vi = end_i / VEC_SIZE;

    // Local scan
    for (uint32_t i = start_vi; i < end_vi; ++i) {
        VecData v = vx[i];
        #pragma unroll
        for (uint32_t vi = 0; vi < VEC_SIZE; ++vi) {
            accumulator = Op::combine(accumulator, v.elements[vi]);
            v.elements[vi] = accumulator;
        }
        // Output to memory
        if constexpr (DO_FIX) { vout[i] = v; }
    }
    // Handle vector tail
    const uint32_t start_scalar_i = end_vi * VEC_SIZE;
    for (uint32_t i = start_scalar_i; i < end_i; ++i) {
        accumulator = Op::combine(accumulator, x[i]);
        if constexpr (DO_FIX) { out[i] = accumulator; }
    }
    return accumulator;
}

template <typename Op, uint32_t VEC_SIZE, bool DO_FIX>
__device__ typename Op::Data warp_scan(
    size_t n, typename Op::Data const *x, typename Op::Data *out, // Work dimensions
    typename Op::Data seed // Seed for thread 0
) {
    using Data = typename Op::Data;

    // Divide x across the threads
    const uint32_t thread_idx = threadIdx.x % 32;
    const uint32_t n_per_thread = ((n / VEC_SIZE) / 32) * VEC_SIZE; // Aligns to vector size
    const uint32_t start_i = thread_idx * n_per_thread;
    const uint32_t end_i = start_i + n_per_thread;

    // Local scan
    Data accumulator = (thread_idx == 0) ? seed : Op::identity();
    accumulator = thread_local_scan<Op, VEC_SIZE, false>(n, x, out, start_i, end_i, accumulator);
    __syncwarp();

    // Hierarchical scan on endpoints
    accumulator = warp_local_scan<Op>(accumulator);

    if constexpr (DO_FIX) {
        // Shuffle accumulators
        accumulator = shfl_up_any(accumulator, 1);
        accumulator = (thread_idx >= 1) ? accumulator : seed;

        // Local scan fix
        accumulator = thread_local_scan<Op, VEC_SIZE, true>(n, x, out, start_i, end_i, accumulator);

        // Handle warp tail
        if (thread_idx == 31) {
            for (uint32_t i = end_i; i < end_i + (n - 32 * n_per_thread); ++i) {
                accumulator = Op::combine(accumulator, x[i]);
                out[i] = accumulator;
            }
        }
    } else {
        // Handle warp tail
        if (thread_idx == 31) {
            for (uint32_t i = end_i; i < end_i + (n - 32 * n_per_thread); ++i) {
                accumulator = Op::combine(accumulator, x[i]);
            }
        }
    }
    return accumulator;
}

template <typename Op, uint32_t VEC_SIZE, bool DO_FIX>
__device__ void warp_scan_handler(
    size_t n, typename Op::Data const *x, typename Op::Data *out, // Work dimensions
    typename Op::Data seed // Seed for thread 0
) {
    using Data = typename Op::Data;

    // Divide x into blocks of 32 * VEC_SIZE and pass to warp scan
    const uint32_t block_size = 32 * VEC_SIZE;
    const uint32_t num_blocks = max((uint32_t)n / block_size, 1u);
    const uint32_t thread_idx = threadIdx.x % 32;

    for (uint32_t idx = 0; idx < num_blocks; ++idx) {
        // Move buffers
        Data const *bx = x + idx * block_size;
        Data *bout = out + idx * block_size;

        // On the last block process whatever is left
        const uint32_t current_block_size = (idx == num_blocks - 1) ? n - idx * block_size : block_size;

        // Call warp scan
        seed = warp_scan<Op, VEC_SIZE, DO_FIX>(current_block_size, bx, bout, seed);
        __syncwarp();

        // For the next block, use the seed from the last thread of this block
        seed = shfl_down_any(seed, 31 - threadIdx.x);
    }

    if constexpr (!DO_FIX) {
        // Only output last accumulator to memory
        if (thread_idx == 31) {
            *out = seed;
        }
    }
}

// 3-Kernel Parallel Algorithm
template <typename Op, uint32_t VEC_SIZE, bool DO_FIX>
__global__ void local_scan(size_t n, typename Op::Data const *x, typename Op::Data *out, typename Op::Data *seed) {
    using Data = typename Op::Data;
    // Thread block info
    const uint32_t num_sm = gridDim.x;
    const uint32_t num_warp = blockDim.x / 32;
    const uint32_t block_idx = blockIdx.x;
    const uint32_t warp_idx = threadIdx.x / 32;

    // Divide x across the SMs
    uint32_t n_per_sm = ((n / VEC_SIZE) / num_sm) * VEC_SIZE; // Aligns to vector size
    Data const *sm_x = x + block_idx * n_per_sm;
    Data *sm_out = out + block_idx * n_per_sm;
    Data *sm_seed = seed + block_idx * num_warp;

    // Handle SM tail
    n_per_sm += (block_idx == num_sm - 1) ? n - num_sm * n_per_sm : 0;

    // Divide sm_x across the warps
    uint32_t n_per_warp = ((n_per_sm / VEC_SIZE) / num_warp) * VEC_SIZE;
    Data const *warp_x = sm_x + warp_idx * n_per_warp;
    Data *warp_out = sm_out + warp_idx * n_per_warp;
    Data *warp_seed = sm_seed + warp_idx;

    // Handle warp tail
    n_per_warp += (warp_idx == num_warp - 1) ? n_per_sm - num_warp * n_per_warp : 0;

    // Call warp scan
    if constexpr (DO_FIX) {
        // Each chunk gets the previous seed
        Data seed_val = (block_idx == 0 && warp_idx == 0) ? Op::identity() : *(warp_seed - 1);
        warp_scan_handler<Op, VEC_SIZE, true>(n_per_warp, warp_x, warp_out, seed_val);
    } else {
        warp_scan_handler<Op, VEC_SIZE, false>(n_per_warp, warp_x, warp_seed, Op::identity());
    }
}
template <typename Op, uint32_t VEC_SIZE>
__global__ void hierarchical_scan(size_t n, typename Op::Data const *x, typename Op::Data *out) {
    warp_scan<Op, VEC_SIZE, true>(n, x, out, Op::identity());
}

template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    typename Op::Data *x, // pointer to GPU memory
    void *workspace       // pointer to GPU memory
) {
    using Data = typename Op::Data;

    // Use the workspace as scratch for seeds
    Data *seed = reinterpret_cast<Data*>(workspace);

    // Thread block dimensions
    constexpr uint32_t B = 48;
    constexpr uint32_t W = 8; // Tuning parameter
    constexpr uint32_t T = 32;

    // Set vector size
    if constexpr (sizeof(Data) > 16) {
        return nullptr;
    }
    constexpr uint32_t VS = 16 / sizeof(Data);

    // Memory
    local_scan<Op, VS, false><<<B, W*T>>>(n, x, x, seed);
    hierarchical_scan<Op, VS><<<1, T>>>(B*W, seed, seed); // Use only 1 SM and 1 warp for the small hierarchical scan
    local_scan<Op, VS, true><<<B, W*T>>>(n, x, x, seed);

    return x;
}

} // namespace scan_gpu

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace rle_gpu {

template <typename Op>
__global__ void create_flag_array(size_t n, char const *raw, typename Op::Data *flag_arr) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        flag_arr[0] = create_rle_data(raw[0], 1);
    }
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < n; idx += gridDim.x * blockDim.x) {
        flag_arr[idx] = create_rle_data(raw[idx], (raw[idx] != raw[idx - 1]));
    }
}

template <typename Op>
__global__ void extract_data(size_t n, typename Op::Data *flag_arr, RleData2 *out) {
    using Data = typename Op::Data;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        Data item = flag_arr[0];
        out[0] = create_rle_data2(item.val, 0);
    }
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < n; idx += gridDim.x * blockDim.x) {
        Data prev = flag_arr[idx - 1];
        Data curr = flag_arr[idx];
        if (prev.count != curr.count) {
            out[curr.count - 1] = create_rle_data2(curr.val, idx);
        }
    }
}

__global__ void extract_compressed_data(
    uint32_t raw_count,
    uint32_t n, RleData2 *cd_out,
    char *compressed_data, uint32_t *compressed_lengths
) {
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n - 1; idx += gridDim.x * blockDim.x) {
        RleData2 curr = cd_out[idx];
        RleData2 next = cd_out[idx + 1];
        compressed_data[idx] = curr.val;
        compressed_lengths[idx] = next.count - curr.count;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        RleData2 curr = cd_out[n - 1];
        compressed_data[n - 1] = curr.val;
        compressed_lengths[n - 1] = raw_count - curr.count;
    }
}

// Returns desired size of scratch buffer in bytes.
size_t get_workspace_size(uint32_t raw_count) {
    using Data = typename SumOp::Data;
    const size_t scan_size = 48 * 8 * sizeof(Data);
    const size_t flag_arr_size = (16778294) * sizeof(Data);
    const size_t compressed_data_size = (16778294) * sizeof(RleData2);
    return scan_size + flag_arr_size + compressed_data_size;
}

// 'launch_rle_compress'
//
// Input:
//
//   'raw_count': Number of bytes in the input buffer 'raw'.
//
//   'raw': Uncompressed bytes in GPU memory.
//
//   'workspace': Scratch buffer in GPU memory. The size of the scratch buffer
//   in bytes is determined by 'get_workspace_size'.
//
// Output:
//
//   Returns: 'compressed_count', the number of runs in the compressed data.
//
//   'compressed_data': Output buffer of size 'raw_count' in GPU memory. The
//   function should fill the first 'compressed_count' bytes of this buffer
//   with the compressed data.
//
//   'compressed_lengths': Output buffer of size 'raw_count' in GPU memory. The
//   function should fill the first 'compressed_count' integers in this buffer
//   with the lengths of the runs in the compressed data.
//
uint32_t launch_rle_compress(
    uint32_t raw_count,
    char const *raw,             // pointer to GPU buffer
    void *workspace,             // pointer to GPU buffer
    char *compressed_data,       // pointer to GPU buffer
    uint32_t *compressed_lengths // pointer to GPU buffer
) {
    using Data = typename SumOp::Data;

    // Partition the workspace
    void *seed = workspace;
    Data *flag = reinterpret_cast<Data*>(seed) + 48 * 8;
    RleData2 *cd_out = reinterpret_cast<RleData2*>((flag + (16778294)));

    // Create flag array
    create_flag_array<SumOp><<<48, 32*32>>>(raw_count, raw, flag);

    // Run scan of raw
    scan_gpu::launch_scan<SumOp>(raw_count, flag, seed);

    // Launch a kernel to extract the data
    extract_data<SumOp><<<48, 8*32>>>(raw_count, flag, cd_out);

    // Extract the compressed count
    Data last_run;
    CUDA_CHECK(cudaMemcpy(&last_run, &flag[raw_count - 1], sizeof(Data), cudaMemcpyDeviceToHost));
    uint32_t compressed_count = last_run.count;

    // Launch a kernel to extract the compressed data and lengths
    extract_compressed_data<<<48, 8*32>>>(raw_count, compressed_count, cd_out, compressed_data, compressed_lengths);

    return compressed_count;
}

} // namespace rle_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Results {
    double time_ms;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

Results run_config(Mode mode, std::vector<char> const &raw) {
    // Allocate buffers
    size_t workspace_size = rle_gpu::get_workspace_size(raw.size());
    char *raw_gpu;
    void *workspace;
    char *compressed_data_gpu;
    uint32_t *compressed_lengths_gpu;
    CUDA_CHECK(cudaMalloc(&raw_gpu, raw.size()));
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    CUDA_CHECK(cudaMalloc(&compressed_data_gpu, raw.size()));
    CUDA_CHECK(cudaMalloc(&compressed_lengths_gpu, raw.size() * sizeof(uint32_t)));

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(raw_gpu, raw.data(), raw.size(), cudaMemcpyHostToDevice));

    auto reset = [&]() {
        CUDA_CHECK(cudaMemset(compressed_data_gpu, 0, raw.size()));
        CUDA_CHECK(cudaMemset(compressed_lengths_gpu, 0, raw.size() * sizeof(uint32_t)));
    };

    auto f = [&]() {
        rle_gpu::launch_rle_compress(
            raw.size(),
            raw_gpu,
            workspace,
            compressed_data_gpu,
            compressed_lengths_gpu);
    };

    // Test correctness
    reset();
    uint32_t compressed_count = rle_gpu::launch_rle_compress(
        raw.size(),
        raw_gpu,
        workspace,
        compressed_data_gpu,
        compressed_lengths_gpu);
    std::vector<char> compressed_data(compressed_count);
    std::vector<uint32_t> compressed_lengths(compressed_count);
    CUDA_CHECK(cudaMemcpy(
        compressed_data.data(),
        compressed_data_gpu,
        compressed_count,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        compressed_lengths.data(),
        compressed_lengths_gpu,
        compressed_count * sizeof(uint32_t),
        cudaMemcpyDeviceToHost));

    std::vector<char> compressed_data_expected;
    std::vector<uint32_t> compressed_lengths_expected;
    rle_compress_cpu(
        raw.size(),
        raw.data(),
        compressed_data_expected,
        compressed_lengths_expected);

    bool correct = true;
    if (compressed_count != compressed_data_expected.size()) {
        printf("Mismatch in compressed count:\n");
        printf("  Expected: %zu\n", compressed_data_expected.size());
        printf("  Actual:   %u\n", compressed_count);
        correct = false;
    }
    if (correct) {
        for (size_t i = 0; i < compressed_data_expected.size(); i++) {
            if (compressed_data[i] != compressed_data_expected[i]) {
                printf("Mismatch in compressed data at index %zu:\n", i);
                printf(
                    "  Expected: 0x%02x\n",
                    static_cast<unsigned char>(compressed_data_expected[i]));
                printf(
                    "  Actual:   0x%02x\n",
                    static_cast<unsigned char>(compressed_data[i]));
                correct = false;
                break;
            }
            if (compressed_lengths[i] != compressed_lengths_expected[i]) {
                printf("Mismatch in compressed lengths at index %zu:\n", i);
                printf("  Expected: %u\n", compressed_lengths_expected[i]);
                printf("  Actual:   %u\n", compressed_lengths[i]);
                correct = false;
                break;
            }
        }
    }
    if (!correct) {
        if (raw.size() <= 1024) {
            printf("\nInput:\n");
            for (size_t i = 0; i < raw.size(); i++) {
                printf("  [%4zu] = 0x%02x\n", i, static_cast<unsigned char>(raw[i]));
            }
            printf("\nExpected:\n");
            for (size_t i = 0; i < compressed_data_expected.size(); i++) {
                printf(
                    "  [%4zu] = data: 0x%02x, length: %u\n",
                    i,
                    static_cast<unsigned char>(compressed_data_expected[i]),
                    compressed_lengths_expected[i]);
            }
            printf("\nActual:\n");
            if (compressed_data.size() == 0) {
                printf("  (empty)\n");
            }
            for (size_t i = 0; i < compressed_data.size(); i++) {
                printf(
                    "  [%4zu] = data: 0x%02x, length: %u\n",
                    i,
                    static_cast<unsigned char>(compressed_data[i]),
                    compressed_lengths[i]);
            }
        }
        exit(1);
    }

    if (mode == Mode::TEST) {
        return {};
    }

    // Benchmark
    double target_time_ms = 1000.0;
    double time_ms = benchmark_ms(target_time_ms, reset, f);

    // Cleanup
    CUDA_CHECK(cudaFree(raw_gpu));
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(compressed_data_gpu));
    CUDA_CHECK(cudaFree(compressed_lengths_gpu));

    return {time_ms};
}

template <typename Rng> std::vector<char> generate_test_data(uint32_t size, Rng &rng) {
    auto random_byte = std::uniform_int_distribution<int32_t>(
        std::numeric_limits<char>::min(),
        std::numeric_limits<char>::max());
    constexpr uint32_t alphabet_size = 4;
    auto alphabet = std::vector<char>();
    for (uint32_t i = 0; i < alphabet_size; i++) {
        alphabet.push_back(random_byte(rng));
    }
    auto random_symbol = std::uniform_int_distribution<uint32_t>(0, alphabet_size - 1);
    auto data = std::vector<char>();
    for (uint32_t i = 0; i < size; i++) {
        data.push_back(alphabet.at(random_symbol(rng)));
    }
    return data;
}

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);

    auto test_sizes = std::vector<uint32_t>{
        16,
        10,
        128,
        100,
        1 << 10,
        1000,
        1 << 20,
        1'000'000,
        16 << 20,
    };

    printf("Correctness:\n\n");
    for (auto test_size : test_sizes) {
        auto raw = generate_test_data(test_size, rng);
        printf("  Testing compression for size %u\n", test_size);
        run_config(Mode::TEST, raw);
        printf("  OK\n\n");
    }

    auto test_data_search_paths = std::vector<std::string>{".", "/"};
    std::string test_data_path;
    for (auto test_data_search_path : test_data_search_paths) {
        auto candidate_path = test_data_search_path + "/rle_raw.bmp";
        if (std::filesystem::exists(candidate_path)) {
            test_data_path = candidate_path;
            break;
        }
    }
    if (test_data_path.empty()) {
        printf("Could not find test data file.\n");
        exit(1);
    }

    auto raw = std::vector<char>();
    {
        auto file = std::ifstream(test_data_path, std::ios::binary);
        if (!file) {
            printf("Could not open test data file '%s'.\n", test_data_path.c_str());
            exit(1);
        }
        file.seekg(0, std::ios::end);
        raw.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(raw.data(), raw.size());
    }

    printf("Performance:\n\n");
    printf("  Testing compression on file 'rle_raw.bmp' (size %zu)\n", raw.size());
    auto results = run_config(Mode::BENCHMARK, raw);
    printf("  Time: %.2f ms\n", results.time_ms);

    return 0;
}