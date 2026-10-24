#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <xrt/experimental/xrt_system.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace fs = std::filesystem;

static constexpr int64_t GGML_XDNA_Q4_0_MATVEC_MAX_K = 8960;
static constexpr int64_t GGML_XDNA_Q4_0_MATVEC_MAX_M = 151936;
static constexpr int64_t GGML_XDNA_Q4_0_MATVEC_TILE_K = 256;
static constexpr int64_t GGML_XDNA_Q4_0_MATVEC_TILE_M = 16;
static constexpr int64_t GGML_XDNA_Q4_0_MATVEC_N = 1;
static constexpr size_t GGML_XDNA_Q4_0_BLOCK_SIZE = 32;
static constexpr size_t GGML_XDNA_Q4_0_BLOCK_BYTES = 18;

struct ggml_backend_xdna_device_context {
    unsigned int xrt_device;
    std::string name;
    std::string description;
    ggml_backend_buffer_type buffer_type = {};
};

struct ggml_backend_xdna_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

struct ggml_backend_xdna_buffer_context {
    xrt::bo bo;
    uint8_t * mapped;
};

struct ggml_backend_xdna_backend_context {
    explicit ggml_backend_xdna_backend_context(unsigned int index) : xrt_device(index) {}

    unsigned int xrt_device;
    std::mutex mutex;
    std::unique_ptr<xrt::device> device;
    struct kernel_context {
        std::unique_ptr<xrt::xclbin> xclbin;
        std::unique_ptr<xrt::hw_context> hw_context;
        std::unique_ptr<xrt::kernel> kernel;
        std::unique_ptr<xrt::bo> instructions_bo;
        size_t instructions_size = 0;
    };
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<kernel_context>> kernels;
};

static ggml_backend_buffer_t ggml_backend_xdna_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
static bool ggml_backend_xdna_supports_q4_0_matvec(const ggml_tensor * op);
static void ggml_backend_xdna_compute_q4_0_matvec(ggml_backend_xdna_backend_context * ctx, ggml_tensor * op);

static bool ggml_backend_xdna_contains(const std::string & value, const char * token) {
    std::string upper = value;
    for (char & c : upper) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return upper.find(token) != std::string::npos;
}

static bool ggml_backend_xdna_is_device(const std::string & name, const std::string & platform) {
    return ggml_backend_xdna_contains(name, "XDNA") ||
           ggml_backend_xdna_contains(name, "NPU") ||
           ggml_backend_xdna_contains(name, "RYZEN AI") ||
           ggml_backend_xdna_contains(platform, "XDNA") ||
           ggml_backend_xdna_contains(platform, "NPU") ||
           ggml_backend_xdna_contains(platform, "RYZEN AI");
}

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "XDNA";
}

static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_xdna_reg_context * ctx = static_cast<ggml_backend_xdna_reg_context *>(reg->context);
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_xdna_reg_context * ctx = static_cast<ggml_backend_xdna_reg_context *>(reg->context);
    return index < ctx->devices.size() ? ctx->devices[index] : nullptr;
}

static const ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name          = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count  = */ ggml_backend_xdna_reg_get_device_count,
    /* .get_device        = */ ggml_backend_xdna_reg_get_device,
    /* .get_proc_address  = */ nullptr,
};

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_xdna_device_context * ctx = static_cast<ggml_backend_xdna_device_context *>(dev->context);
    return ctx->name.c_str();
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_xdna_device_context * ctx = static_cast<ggml_backend_xdna_device_context *>(dev->context);
    return ctx->description.c_str();
}

static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free = 0;
    *total = 0;
}

static const char * ggml_backend_xdna_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_xdna_device_context * ctx = static_cast<ggml_backend_xdna_device_context *>(buft->context);
    return ctx->name.c_str();
}

static size_t ggml_backend_xdna_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
}

static size_t ggml_backend_xdna_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return std::numeric_limits<size_t>::max();
}

static size_t ggml_backend_xdna_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static bool ggml_backend_xdna_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static void ggml_backend_xdna_buffer_sync(
        ggml_backend_xdna_buffer_context * ctx,
        xclBOSyncDirection direction,
        size_t size,
        size_t offset) {
    if (size == 0) {
        return;
    }

    try {
        ctx->bo.sync(direction, size, offset);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: XRT buffer synchronization failed: %s\n", __func__, e.what());
        GGML_ABORT("XDNA XRT buffer synchronization failed");
    }
}

static size_t ggml_backend_xdna_buffer_tensor_offset(
        ggml_backend_buffer_t buffer,
        const ggml_tensor * tensor) {
    ggml_backend_xdna_buffer_context * ctx = static_cast<ggml_backend_xdna_buffer_context *>(buffer->context);
    const uintptr_t base = reinterpret_cast<uintptr_t>(ctx->mapped);
    const uintptr_t data = reinterpret_cast<uintptr_t>(tensor->data);
    GGML_ASSERT(data >= base);

    const size_t offset = static_cast<size_t>(data - base);
    GGML_ASSERT(offset <= buffer->size);
    return offset;
}

static void ggml_backend_xdna_buffer_set_tensor_2d(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size,
        size_t n_copies,
        size_t stride_tensor,
        size_t stride_data) {
    if (size == 0 || n_copies == 0) {
        return;
    }

    ggml_backend_xdna_buffer_context * ctx = static_cast<ggml_backend_xdna_buffer_context *>(buffer->context);
    const size_t tensor_offset = ggml_backend_xdna_buffer_tensor_offset(buffer, tensor);
    GGML_ASSERT(offset <= buffer->size - tensor_offset);
    const size_t buffer_offset = tensor_offset + offset;
    const size_t transfer_size = (n_copies - 1) * stride_tensor + size;
    GGML_ASSERT(transfer_size <= buffer->size - buffer_offset);

    uint8_t * dst = ctx->mapped + buffer_offset;
    const uint8_t * src = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n_copies; i++) {
        std::memcpy(dst + i * stride_tensor, src + i * stride_data, size);
    }
    ggml_backend_xdna_buffer_sync(ctx, XCL_BO_SYNC_BO_TO_DEVICE, transfer_size, buffer_offset);
}

static void ggml_backend_xdna_buffer_get_tensor_2d(
        ggml_backend_buffer_t buffer,
        const ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size,
        size_t n_copies,
        size_t stride_tensor,
        size_t stride_data) {
    if (size == 0 || n_copies == 0) {
        return;
    }

    ggml_backend_xdna_buffer_context * ctx = static_cast<ggml_backend_xdna_buffer_context *>(buffer->context);
    const size_t tensor_offset = ggml_backend_xdna_buffer_tensor_offset(buffer, tensor);
    GGML_ASSERT(offset <= buffer->size - tensor_offset);
    const size_t buffer_offset = tensor_offset + offset;
    const size_t transfer_size = (n_copies - 1) * stride_tensor + size;
    GGML_ASSERT(transfer_size <= buffer->size - buffer_offset);

    ggml_backend_xdna_buffer_sync(ctx, XCL_BO_SYNC_BO_FROM_DEVICE, transfer_size, buffer_offset);
    const uint8_t * src = ctx->mapped + buffer_offset;
    uint8_t * dst = static_cast<uint8_t *>(data);
    for (size_t i = 0; i < n_copies; i++) {
        std::memcpy(dst + i * stride_data, src + i * stride_tensor, size);
    }
}

static void ggml_backend_xdna_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    delete static_cast<ggml_backend_xdna_buffer_context *>(buffer->context);
}

static void * ggml_backend_xdna_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_xdna_buffer_context * ctx = static_cast<ggml_backend_xdna_buffer_context *>(buffer->context);
    return ctx->mapped;
}

static void ggml_backend_xdna_buffer_memset_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        uint8_t value,
        size_t offset,
        size_t size) {
    if (size == 0) {
        return;
    }

    ggml_backend_xdna_buffer_context * ctx = static_cast<ggml_backend_xdna_buffer_context *>(buffer->context);
    const size_t tensor_offset = ggml_backend_xdna_buffer_tensor_offset(buffer, tensor);
    GGML_ASSERT(offset <= buffer->size - tensor_offset);
    const size_t buffer_offset = tensor_offset + offset;
    GGML_ASSERT(size <= buffer->size - buffer_offset);

    std::memset(ctx->mapped + buffer_offset, value, size);
    ggml_backend_xdna_buffer_sync(ctx, XCL_BO_SYNC_BO_TO_DEVICE, size, buffer_offset);
}

static void ggml_backend_xdna_buffer_set_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    ggml_backend_xdna_buffer_set_tensor_2d(buffer, tensor, data, offset, size, 1, 0, 0);
}

static void ggml_backend_xdna_buffer_get_tensor(
        ggml_backend_buffer_t buffer,
        const ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size) {
    ggml_backend_xdna_buffer_get_tensor_2d(buffer, tensor, data, offset, size, 1, 0, 0);
}

static bool ggml_backend_xdna_is_buffer_type(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.alloc_buffer == ggml_backend_xdna_buffer_type_alloc_buffer;
}

static bool ggml_backend_xdna_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer,
        const ggml_tensor * src,
        ggml_tensor * dst) {
    if (src->buffer == nullptr || !ggml_backend_xdna_is_buffer_type(src->buffer->buft) ||
        !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const size_t size = ggml_nbytes(src);
    if (size == 0) {
        return true;
    }

    ggml_backend_xdna_buffer_context * src_ctx = static_cast<ggml_backend_xdna_buffer_context *>(src->buffer->context);
    ggml_backend_xdna_buffer_context * dst_ctx = static_cast<ggml_backend_xdna_buffer_context *>(buffer->context);
    const size_t src_offset = ggml_backend_xdna_buffer_tensor_offset(src->buffer, src);
    const size_t dst_offset = ggml_backend_xdna_buffer_tensor_offset(buffer, dst);
    GGML_ASSERT(size <= src->buffer->size - src_offset);
    GGML_ASSERT(size <= buffer->size - dst_offset);

    ggml_backend_xdna_buffer_sync(src_ctx, XCL_BO_SYNC_BO_FROM_DEVICE, size, src_offset);
    std::memmove(dst_ctx->mapped + dst_offset, src_ctx->mapped + src_offset, size);
    ggml_backend_xdna_buffer_sync(dst_ctx, XCL_BO_SYNC_BO_TO_DEVICE, size, dst_offset);
    return true;
}

static void ggml_backend_xdna_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_xdna_buffer_context * ctx = static_cast<ggml_backend_xdna_buffer_context *>(buffer->context);
    if (buffer->size == 0) {
        return;
    }

    std::memset(ctx->mapped, value, buffer->size);
    ggml_backend_xdna_buffer_sync(ctx, XCL_BO_SYNC_BO_TO_DEVICE, buffer->size, 0);
}

static fs::path ggml_backend_xdna_module_dir() {
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&ggml_backend_xdna_reg_i), &module)) {
        throw std::runtime_error("failed to locate XDNA backend module");
    }

    std::vector<wchar_t> path(32768);
    const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length >= static_cast<DWORD>(path.size())) {
        throw std::runtime_error("failed to read XDNA backend module path");
    }
    return fs::path(std::wstring(path.data(), length)).parent_path();
#else
    Dl_info info = {};
    if (dladdr(static_cast<const void *>(&ggml_backend_xdna_reg_i), &info) == 0 || info.dli_fname == nullptr) {
        throw std::runtime_error("failed to locate XDNA backend module");
    }
    return fs::path(info.dli_fname).parent_path();
#endif
}

static fs::path ggml_backend_xdna_kernel_dir() {
    const char * override_dir = std::getenv("GGML_XDNA_KERNEL_DIR");
    if (override_dir != nullptr && override_dir[0] != '\0') {
        return fs::path(override_dir);
    }
    return ggml_backend_xdna_module_dir() / "xdna-kernels";
}

static std::vector<char> ggml_backend_xdna_read_file(const fs::path & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("failed to open " + path.string());
    }

    const std::streamoff size = file.tellg();
    constexpr std::streamoff max_file_size = 64 * 1024 * 1024;
    if (size <= 0 || size > max_file_size) {
        throw std::runtime_error("invalid file size for " + path.string());
    }

    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("failed to read " + path.string());
    }
    return data;
}

static ggml_backend_xdna_backend_context::kernel_context & ggml_backend_xdna_load_q4_0_matvec(
        ggml_backend_xdna_backend_context * ctx,
        int64_t m,
        int64_t k) {
    const auto key = std::make_pair(m, k);
    const auto existing = ctx->kernels.find(key);
    if (existing != ctx->kernels.end()) {
        return *existing->second;
    }

    auto kernel_ctx = std::make_unique<ggml_backend_xdna_backend_context::kernel_context>();
    const fs::path artifact_dir = ggml_backend_xdna_kernel_dir() /
            ("q4_0_matvec_" + std::to_string(m) + "x1x" + std::to_string(k));
    const fs::path xclbin_path = artifact_dir / "q4_0_matvec.xclbin";
    const fs::path instructions_path = artifact_dir / "q4_0_matvec.insts.bin";

    if (ctx->device == nullptr) {
        ctx->device = std::make_unique<xrt::device>(ctx->xrt_device);
    }
    auto xclbin = std::make_unique<xrt::xclbin>(xclbin_path.string());
    const xrt::uuid uuid = ctx->device->register_xclbin(*xclbin);
    auto hw_context = std::make_unique<xrt::hw_context>(*ctx->device, uuid);
    auto kernel = std::make_unique<xrt::kernel>(*hw_context, "MLIR_AIE");
    const std::vector<char> instructions = ggml_backend_xdna_read_file(instructions_path);
    auto instructions_bo = std::make_unique<xrt::bo>(
            *ctx->device,
            instructions.size(),
            xrt::bo::flags::cacheable,
            static_cast<xrt::memory_group>(kernel->group_id(1)));

    void * mapped = instructions_bo->map();
    if (mapped == nullptr) {
        throw std::runtime_error("XRT returned a null instruction-buffer mapping");
    }
    std::memcpy(mapped, instructions.data(), instructions.size());
    instructions_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, instructions.size(), 0);

    kernel_ctx->xclbin = std::move(xclbin);
    kernel_ctx->hw_context = std::move(hw_context);
    kernel_ctx->kernel = std::move(kernel);
    kernel_ctx->instructions_bo = std::move(instructions_bo);
    kernel_ctx->instructions_size = instructions.size();
    return *ctx->kernels.emplace(key, std::move(kernel_ctx)).first->second;
}

static xrt::bo ggml_backend_xdna_make_input_bo(xrt::device & device, const void * data, size_t size) {
    xrt::bo buffer(device, size, xrt::bo::flags::host_only, static_cast<xrt::memory_group>(0));
    uint8_t * mapped = buffer.map<uint8_t *>();
    if (mapped == nullptr) {
        throw std::runtime_error("XRT returned a null input-buffer mapping");
    }
    std::memcpy(mapped, data, size);
    buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE, size, 0);
    return buffer;
}

static xrt::bo ggml_backend_xdna_make_output_bo(xrt::device & device, size_t size) {
    xrt::bo buffer(device, size, xrt::bo::flags::host_only, static_cast<xrt::memory_group>(0));
    if (buffer.map() == nullptr) {
        throw std::runtime_error("XRT returned a null output-buffer mapping");
    }
    return buffer;
}

static bool ggml_backend_xdna_supports_q4_0_matvec(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * weights = op->src[0];
    const ggml_tensor * input = op->src[1];
    const int64_t k = weights->ne[0];
    const int64_t m = weights->ne[1];
    const bool known_shape =
            (k == 256 && m == 16) ||
            (k == 1536 && (m == 256 || m == 1536 || m == 8960)) ||
            (k == 8960 && m == 1536);
    return weights->type == GGML_TYPE_Q4_0 &&
           input->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           known_shape &&
           k > 0 &&
           k <= GGML_XDNA_Q4_0_MATVEC_MAX_K &&
           k % GGML_XDNA_Q4_0_MATVEC_TILE_K == 0 &&
           m > 0 &&
           m <= GGML_XDNA_Q4_0_MATVEC_MAX_M &&
           m % GGML_XDNA_Q4_0_MATVEC_TILE_M == 0 &&
           weights->ne[2] == 1 &&
           weights->ne[3] == 1 &&
           input->ne[0] == k &&
           input->ne[1] == 1 &&
           input->ne[2] == 1 &&
           input->ne[3] == 1 &&
           op->ne[0] == m &&
           op->ne[1] == 1 &&
           op->ne[2] == 1 &&
           op->ne[3] == 1 &&
           ggml_is_contiguous(weights) &&
           ggml_is_contiguous(input) &&
           ggml_is_contiguous(op) &&
           ggml_get_op_params_i32(op, 0) == GGML_PREC_DEFAULT &&
           ggml_get_op_params_i32(op, 1) == 0;
}

static void ggml_backend_xdna_compute_q4_0_matvec(ggml_backend_xdna_backend_context * ctx, ggml_tensor * op) {
    const ggml_tensor * weights = op->src[0];
    const ggml_tensor * input = op->src[1];
    const size_t weights_size = ggml_nbytes(weights);
    const size_t input_size = ggml_nbytes(input);
    const size_t output_size = ggml_nbytes(op);
    const size_t k = static_cast<size_t>(weights->ne[0]);
    const size_t m = static_cast<size_t>(weights->ne[1]);
    const size_t expected_weights_size = m * (k / GGML_XDNA_Q4_0_BLOCK_SIZE) * GGML_XDNA_Q4_0_BLOCK_BYTES;
    const size_t expected_input_size = k * sizeof(float);
    const size_t expected_output_size = m * sizeof(float);
    if (weights_size != expected_weights_size || input_size != expected_input_size || output_size != expected_output_size) {
        throw std::runtime_error("unexpected Q4_0 MUL_MAT tensor size");
    }

    ggml_backend_xdna_backend_context::kernel_context & kernel_ctx =
            ggml_backend_xdna_load_q4_0_matvec(ctx, static_cast<int64_t>(m), static_cast<int64_t>(k));
    std::vector<uint8_t> weights_data(weights_size);
    std::vector<float> input_data(k);
    std::vector<float> output_data(m);
    ggml_backend_tensor_get(weights, weights_data.data(), 0, weights_size);
    ggml_backend_tensor_get(input, input_data.data(), 0, input_size);
    xrt::bo weights_bo = ggml_backend_xdna_make_input_bo(*ctx->device, weights_data.data(), weights_size);
    xrt::bo input_bo = ggml_backend_xdna_make_input_bo(*ctx->device, input_data.data(), input_size);
    xrt::bo output_bo = ggml_backend_xdna_make_output_bo(*ctx->device, output_size);
    xrt::run run = (*kernel_ctx.kernel)(
            3,
            *kernel_ctx.instructions_bo,
            kernel_ctx.instructions_size,
            weights_bo,
            input_bo,
            output_bo);
    const auto run_state = run.wait();
    if (run_state != ERT_CMD_STATE_COMPLETED) {
        throw std::runtime_error(
                "XRT Q4_0 MUL_MAT kernel did not complete (state " +
                std::to_string(static_cast<int>(run_state)) + ")");
    }

    output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, output_size, 0);
    const float * mapped = output_bo.map<float *>();
    if (mapped == nullptr) {
        throw std::runtime_error("XRT returned a null output-buffer mapping after execution");
    }
    std::memcpy(output_data.data(), mapped, output_size);
    ggml_backend_tensor_set(op, output_data.data(), 0, output_size);
}

static const ggml_backend_buffer_i ggml_backend_xdna_buffer_i = {
    /* .free_buffer     = */ ggml_backend_xdna_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_xdna_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ ggml_backend_xdna_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_xdna_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_xdna_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_xdna_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_xdna_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_xdna_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_xdna_buffer_clear,
    /* .reset           = */ nullptr,
};

static ggml_backend_buffer_t ggml_backend_xdna_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft,
        size_t size) {
    ggml_backend_xdna_device_context * dev_ctx = static_cast<ggml_backend_xdna_device_context *>(buft->context);

    try {
        xrt::device device(dev_ctx->xrt_device);
        xrt::bo bo(device, size, xrt::bo::flags::host_only, static_cast<xrt::memory_group>(0));
        uint8_t * mapped = bo.map<uint8_t *>();
        if (mapped == nullptr) {
            GGML_LOG_ERROR("%s: XRT returned a null mapping for a %zu-byte XDNA buffer\n", __func__, size);
            return nullptr;
        }

        const size_t alignment = ggml_backend_xdna_buffer_type_get_alignment(buft);
        if (reinterpret_cast<uintptr_t>(mapped) % alignment != 0) {
            GGML_LOG_ERROR("%s: XRT mapping for XDNA device %u is not aligned to %zu bytes\n",
                    __func__, dev_ctx->xrt_device, alignment);
            return nullptr;
        }

        std::unique_ptr<ggml_backend_xdna_buffer_context> ctx(
                new ggml_backend_xdna_buffer_context { std::move(bo), mapped });
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, ggml_backend_xdna_buffer_i, ctx.get(), size);
        ctx.release();
        return buffer;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to allocate %zu-byte XDNA buffer on XRT device %u: %s\n",
                __func__, size, dev_ctx->xrt_device, e.what());
        return nullptr;
    }
}

static const ggml_backend_buffer_type_i ggml_backend_xdna_buffer_type_i = {
    /* .get_name         = */ ggml_backend_xdna_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_xdna_buffer_type_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ ggml_backend_xdna_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_xdna_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_xdna_buffer_type_get_alloc_size,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ ggml_backend_xdna_buffer_type_is_host,
};

static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna_device_get_name(dev);
    props->description = ggml_backend_xdna_device_get_description(dev);
    props->type        = ggml_backend_xdna_device_get_type(dev);
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id   = nullptr;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ true,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ true,
    };
}

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "XDNA";
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    delete static_cast<ggml_backend_xdna_backend_context *>(backend->context);
    delete backend;
}

static enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    if (cgraph->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    ggml_backend_xdna_backend_context * ctx = static_cast<ggml_backend_xdna_backend_context *>(backend->context);
    std::lock_guard<std::mutex> lock(ctx->mutex);
    try {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * op = cgraph->nodes[i];
            if (op->op == GGML_OP_NONE) {
                continue;
            }
            if (!ggml_backend_xdna_supports_q4_0_matvec(op)) {
                GGML_LOG_ERROR("%s: unsupported XDNA operation %s\n", __func__, ggml_op_name(op->op));
                return GGML_STATUS_FAILED;
            }
            ggml_backend_xdna_compute_q4_0_matvec(ctx, op);
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: XDNA graph execution failed: %s\n", __func__, e.what());
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_xdna_i = {
    /* .get_name                = */ ggml_backend_xdna_get_name,
    /* .free                    = */ ggml_backend_xdna_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_xdna_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = { 0x2d, 0x6b, 0x9f, 0x42, 0x31, 0xf0, 0x4a, 0xa7, 0x8b, 0x3c, 0xf1, 0x0e, 0x4d, 0xa8, 0xc2, 0x17 };
    return &guid;
}

static ggml_backend_t ggml_backend_xdna_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_xdna_device_context * dev_ctx = static_cast<ggml_backend_xdna_device_context *>(dev->context);
    std::unique_ptr<ggml_backend_xdna_backend_context> ctx(new ggml_backend_xdna_backend_context(dev_ctx->xrt_device));
    return new ggml_backend {
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ ggml_backend_xdna_i,
        /* .device  = */ dev,
        /* .context = */ ctx.release(),
    };
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_xdna_device_context * ctx = static_cast<ggml_backend_xdna_device_context *>(dev->context);
    return &ctx->buffer_type;
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (op != nullptr && op->op == GGML_OP_NONE) {
        return true;
    }
    return ggml_backend_xdna_supports_q4_0_matvec(op);
}

static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_cpu_buffer_type() ||
           (ggml_backend_xdna_is_buffer_type(buft) && buft->device == dev);
}

static const ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_xdna_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_xdna_reg_context * ctx = new ggml_backend_xdna_reg_context;

            unsigned int n_devices = 0;
            try {
                n_devices = xrt::system::enumerate_devices();
            } catch (const std::exception & e) {
                GGML_LOG_ERROR("%s: failed to enumerate XRT devices: %s\n", __func__, e.what());
            }

            for (unsigned int i = 0; i < n_devices; i++) {
                std::string name;
                std::string platform;
                try {
                    xrt::device xrt_device(i);
                    name = xrt_device.get_info<xrt::info::device::name>();
                    try {
                        platform = xrt_device.get_info<xrt::info::device::platform>();
                    } catch (const std::exception & e) {
                        GGML_LOG_DEBUG("%s: failed to read platform for XRT device %u: %s\n", __func__, i, e.what());
                    }
                } catch (const std::exception & e) {
                    GGML_LOG_WARN("%s: failed to inspect XRT device %u: %s\n", __func__, i, e.what());
                    continue;
                }

                if (!ggml_backend_xdna_is_device(name, platform)) {
                    GGML_LOG_DEBUG("%s: skipping non-XDNA XRT device %u (%s, %s)\n", __func__, i, name.c_str(), platform.c_str());
                    continue;
                }

                ggml_backend_xdna_device_context * dev_ctx = new ggml_backend_xdna_device_context;
                dev_ctx->xrt_device = i;
                dev_ctx->name = "XDNA" + std::to_string(ctx->devices.size());
                dev_ctx->description = platform.empty() ? name : name + " (" + platform + ")";

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_xdna_device_i,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx,
                };
                dev_ctx->buffer_type = {
                    /* .iface   = */ ggml_backend_xdna_buffer_type_i,
                    /* .device  = */ dev,
                    /* .context = */ dev_ctx,
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_xdna_reg_i,
                /* .context     = */ ctx,
            };
            initialized = true;
        }
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
