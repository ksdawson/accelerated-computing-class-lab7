#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
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

template <typename Op>
void print_array(
    size_t n,
    typename Op::Data const *x // allowed to be either a CPU or GPU pointer
);

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

template <typename Op>
void scan_cpu(size_t n, typename Op::Data const *x, typename Op::Data *out) {
    using Data = typename Op::Data;
    Data accumulator = Op::identity();
    for (size_t i = 0; i < n; i++) {
        accumulator = Op::combine(accumulator, x[i]);
        out[i] = accumulator;
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

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
__device__ void warp_scan(
    size_t n, typename Op::Data const *x, typename Op::Data *out, // Work dimensions
    typename Op::Data seed // Seed for thread 0
) {
    // Data types
    using Data = typename Op::Data;
    using VecData = Vectorized<Data, VEC_SIZE>;

    // Divide x across the threads
    const uint32_t thread_idx = threadIdx.x % 32;
    const uint32_t n_per_thread = ((n / VEC_SIZE) / 32) * VEC_SIZE; // Aligns to vector size
    const uint32_t start_i = thread_idx * n_per_thread;
    const uint32_t end_i = start_i + n_per_thread;
    const uint32_t start_vi = start_i / VEC_SIZE;
    const uint32_t end_vi = end_i / VEC_SIZE;

    // Vectorize
    VecData const *vx = reinterpret_cast<VecData const *>(x);
    VecData *vout = reinterpret_cast<VecData*>(out);

    // Local scan
    Data accumulator = (thread_idx == 0) ? seed : Op::identity();
    for (uint32_t i = start_vi; i < end_vi; ++i) {
        VecData v = vx[i];
        #pragma unroll
        for (uint32_t vi = 0; vi < VEC_SIZE; ++vi) {
            accumulator = Op::combine(accumulator, v.elements[vi]);
        }
    }
    // Handle vector tail
    const uint32_t start_scalar_i = end_vi * VEC_SIZE;
    for (uint32_t i = start_scalar_i; i < end_i; ++i) {
        accumulator = Op::combine(accumulator, x[i]);
    }
    __syncwarp();

    // Hierarchical scan on endpoints
    accumulator = warp_local_scan<Op>(accumulator);

    if constexpr (DO_FIX) {
        // Shuffle accumulators
        accumulator = shfl_up_any(accumulator, 1);
        accumulator = (thread_idx >= 1) ? accumulator : seed;

        // Local scan fix
        for (uint32_t i = start_vi; i < end_vi; ++i) {
            VecData v = vx[i];
            #pragma unroll
            for (uint32_t vi = 0; vi < VEC_SIZE; ++vi) {
                accumulator = Op::combine(accumulator, v.elements[vi]);
                v.elements[vi] = accumulator;
            }
            // Output to memory
            vout[i] = v;
        }
        // Handle vector tail
        for (uint32_t i = start_scalar_i; i < end_i; ++i) {
            accumulator = Op::combine(accumulator, x[i]);
            out[i] = accumulator;
        }

        // Handle warp tail
        if (thread_idx == 31) {
            for (uint32_t i = end_i; i < end_i + (n - 32 * n_per_thread); ++i) {
                accumulator = Op::combine(accumulator, x[i]);
                out[i] = accumulator;
            }
        }
    } else {
        // Only output last accumulator to memory
        if (thread_idx == 31) {
            // Handle warp tail
            for (uint32_t i = end_i; i < end_i + (n - 32 * n_per_thread); ++i) {
                accumulator = Op::combine(accumulator, x[i]);
            }
            *out = accumulator;
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
        warp_scan<Op, VEC_SIZE, true>(n_per_warp, warp_x, warp_out, seed_val);
    } else {
        warp_scan<Op, VEC_SIZE, false>(n_per_warp, warp_x, warp_seed, Op::identity());
    }
}
template <typename Op, uint32_t VEC_SIZE>
__global__ void hierarchical_scan(size_t n, typename Op::Data const *x, typename Op::Data *out) {
    warp_scan<Op, VEC_SIZE, true>(n, x, out, Op::identity());
}

// Returns desired size of scratch buffer in bytes.
template <typename Op> size_t get_workspace_size(size_t n) {
    using Data = typename Op::Data;
    return 1 * 16 * sizeof(Data); // num sms x num warps
}

// 'launch_scan'
//
// Input:
//
//   'n': Number of elements in the input array 'x'.
//
//   'x': Input array in GPU memory. The 'launch_scan' function is allowed to
//   overwrite the contents of this buffer.
//
//   'workspace': Scratch buffer in GPU memory. The size of the scratch buffer
//   in bytes is determined by 'get_workspace_size<Op>(n)'.
//
// Output:
//
//   Returns a pointer to GPU memory which will contain the results of the scan
//   after all launched kernels have completed. Must be either a pointer to the
//   'x' buffer or to an offset within the 'workspace' buffer.
//
//   The contents of the output array should be "partial reductions" of the
//   input; each element 'i' of the output array should be given by:
//
//     output[i] = Op::combine(x[0], x[1], ..., x[i])
//
//   where 'Op::combine(...)' of more than two arguments is defined in terms of
//   repeatedly combining pairs of arguments. Note that 'Op::combine' is
//   guaranteed to be associative, but not necessarily commutative, so
//
//        Op::combine(a, b, c)              // conceptual notation; not real C++
//     == Op::combine(a, Op::combine(b, c)) // real C++
//     == Op::combine(Op::combine(a, b), c) // real C++
//
//  but we don't necessarily have
//
//    Op::combine(a, b) == Op::combine(b, a) // not true in general!
//
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
    constexpr uint32_t B = 1;
    constexpr uint32_t W = 16;
    constexpr uint32_t T = 32;

    if (sizeof(Data) == 4) {
        local_scan<Op, 4, false><<<B, W*T>>>(n, x, x, seed);
        hierarchical_scan<Op, 4><<<1, T>>>(B*W, seed, seed); // Use only 1 SM and 1 warp for the small hierarchical scan
        local_scan<Op, 4, true><<<B, W*T>>>(n, x, x, seed);
        return x;
    } else if (sizeof(Data) == 8) {
        local_scan<Op, 2, false><<<B, W*T>>>(n, x, x, seed);
        hierarchical_scan<Op, 2><<<1, T>>>(B*W, seed, seed);
        local_scan<Op, 2, true><<<B, W*T>>>(n, x, x, seed);
        return x;
    } else {
        return nullptr;
    }
}

} // namespace scan_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

struct DebugRange {
    uint32_t lo;
    uint32_t hi;

    static constexpr uint32_t INVALID = 0xffffffff;

    static __host__ __device__ __forceinline__ DebugRange invalid() {
        return {INVALID, INVALID};
    }

    __host__ __device__ __forceinline__ bool operator==(const DebugRange &other) const {
        return lo == other.lo && hi == other.hi;
    }

    __host__ __device__ __forceinline__ bool operator!=(const DebugRange &other) const {
        return !(*this == other);
    }

    __host__ __device__ bool is_empty() const { return lo == hi; }

    __host__ __device__ bool is_valid() const { return lo != INVALID; }

    std::string to_string() const {
        if (lo == INVALID) {
            return "INVALID";
        } else {
            return std::to_string(lo) + ":" + std::to_string(hi);
        }
    }
};

struct DebugRangeConcatOp {
    using Data = DebugRange;

    static __host__ __device__ __forceinline__ Data identity() { return {0, 0}; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        if (a.is_empty()) {
            return b;
        } else if (b.is_empty()) {
            return a;
        } else if (a.is_valid() && b.is_valid() && a.hi == b.lo) {
            return {a.lo, b.hi};
        } else {
            return Data::invalid();
        }
    }

    static std::string to_string(Data d) { return d.to_string(); }
};

struct SumOp {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }

    static std::string to_string(Data d) { return std::to_string(d); }
};

constexpr size_t max_print_array_output = 1025;
static thread_local size_t total_print_array_output = 0;

template <typename Op> void print_array(size_t n, typename Op::Data const *x) {
    using Data = typename Op::Data;

    // copy 'x' from device to host if necessary
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, x));
    auto x_host_buf = std::vector<Data>();
    Data const *x_host_ptr = nullptr;
    if (attr.type == cudaMemoryTypeDevice) {
        x_host_buf.resize(n);
        x_host_ptr = x_host_buf.data();
        CUDA_CHECK(
            cudaMemcpy(x_host_buf.data(), x, n * sizeof(Data), cudaMemcpyDeviceToHost));
    } else {
        x_host_ptr = x;
    }

    if (total_print_array_output >= max_print_array_output) {
        return;
    }

    printf("[\n");
    for (size_t i = 0; i < n; i++) {
        auto s = Op::to_string(x_host_ptr[i]);
        printf("  [%zu] = %s,\n", i, s.c_str());
        total_print_array_output++;
        if (total_print_array_output > max_print_array_output) {
            printf("  ... (output truncated)\n");
            break;
        }
    }
    printf("]\n");

    if (total_print_array_output >= max_print_array_output) {
        printf("(Reached maximum limit on 'print_array' output; skipping further calls "
               "to 'print_array')\n");
    }

    total_print_array_output++;
}

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
    double bandwidth_gb_per_sec;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename Op>
Results run_config(Mode mode, std::vector<typename Op::Data> const &x) {
    // Allocate buffers
    using Data = typename Op::Data;
    size_t n = x.size();
    size_t workspace_size = scan_gpu::get_workspace_size<Op>(n);
    Data *x_gpu;
    Data *workspace_gpu;
    CUDA_CHECK(cudaMalloc(&x_gpu, n * sizeof(Data)));
    CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
    CUDA_CHECK(cudaMemcpy(x_gpu, x.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));

    // Test correctness
    auto expected = std::vector<Data>(n);
    scan_cpu<Op>(n, x.data(), expected.data());
    auto out_gpu = scan_gpu::launch_scan<Op>(n, x_gpu, workspace_gpu);
    if (out_gpu == nullptr) {
        printf("'launch_scan' function not yet implemented (returned nullptr)\n");
        exit(1);
    }
    auto actual = std::vector<Data>(n);
    CUDA_CHECK(
        cudaMemcpy(actual.data(), out_gpu, n * sizeof(Data), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        if (actual.at(i) != expected.at(i)) {
            auto actual_str = Op::to_string(actual.at(i));
            auto expected_str = Op::to_string(expected.at(i));
            printf(
                "Mismatch at position %zu: %s != %s\n",
                i,
                actual_str.c_str(),
                expected_str.c_str());
            if (n <= 128) {
                printf("Input:\n");
                print_array<Op>(n, x.data());
                printf("\nExpected:\n");
                print_array<Op>(n, expected.data());
                printf("\nActual:\n");
                print_array<Op>(n, actual.data());
            }
            exit(1);
        }
    }
    if (mode == Mode::TEST) {
        return {0.0, 0.0};
    }

    // Benchmark
    double target_time_ms = 200.0;
    double time_ms = benchmark_ms(
        target_time_ms,
        [&]() {
            CUDA_CHECK(
                cudaMemcpy(x_gpu, x.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
        },
        [&]() { scan_gpu::launch_scan<Op>(n, x_gpu, workspace_gpu); });
    double bytes_processed = n * sizeof(Data) * 2;
    double bandwidth_gb_per_sec = bytes_processed / time_ms / 1e6;

    // Cleanup
    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(workspace_gpu));

    return {time_ms, bandwidth_gb_per_sec};
}

std::vector<DebugRange> gen_debug_ranges(uint32_t n) {
    auto ranges = std::vector<DebugRange>();
    for (uint32_t i = 0; i < n; ++i) {
        ranges.push_back({i, i + 1});
    }
    return ranges;
}

template <typename Rng> std::vector<uint32_t> gen_random_data(Rng &rng, uint32_t n) {
    auto uniform = std::uniform_int_distribution<uint32_t>(0, 100);
    auto data = std::vector<uint32_t>();
    for (uint32_t i = 0; i < n; ++i) {
        data.push_back(uniform(rng));
    }
    return data;
}

template <typename Op, typename GenData>
void run_tests(std::vector<uint32_t> const &sizes, GenData &&gen_data) {
    for (auto size : sizes) {
        auto data = gen_data(size);
        printf("  Testing size %8u\n", size);
        run_config<Op>(Mode::TEST, data);
        printf("  OK\n\n");
    }
}

int main(int argc, char const *const *argv) {
    auto correctness_sizes = std::vector<uint32_t>{
        16,
        10,
        128,
        100,
        1024,
        1000,
        // 1 << 20,
        // 1'000'000,
        // 16 << 20,
        // 64 << 20,
    };

    auto rng = std::mt19937(0xCA7CAFE);

    printf("Correctness:\n\n");
    printf("Testing scan operation: debug range concatenation\n\n");
    run_tests<DebugRangeConcatOp>(correctness_sizes, gen_debug_ranges);
    printf("Testing scan operation: integer sum\n\n");
    run_tests<SumOp>(correctness_sizes, [&](uint32_t n) {
        return gen_random_data(rng, n);
    });

    printf("Performance:\n\n");

    size_t n = 64 << 20;
    auto data = gen_random_data(rng, n);

    printf("Benchmarking scan operation: integer sum, size %zu\n\n", n);

    // Warmup
    run_config<SumOp>(Mode::BENCHMARK, data);
    // Benchmark
    auto results = run_config<SumOp>(Mode::BENCHMARK, data);
    printf("  Time: %.2f ms\n", results.time_ms);
    printf("  Throughput: %.2f GB/s\n", results.bandwidth_gb_per_sec);

    return 0;
}