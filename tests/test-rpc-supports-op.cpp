#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <cinttypes>
#include <cstdio>
#include <utility>
#include <vector>

static bool test_shape(ggml_context * ctx, ggml_backend_dev_t dev, int64_t m, int64_t k, bool expected) {
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, m);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
    const bool supported = ggml_backend_dev_supports_op(dev, output);
    if (supported != expected) {
        fprintf(stderr, "RPC device %s reported (M=%" PRId64 ", K=%" PRId64 ") as %s; expected %s\n",
                ggml_backend_dev_name(dev), m, k, supported ? "supported" : "unsupported",
                expected ? "supported" : "unsupported");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s HOST:PORT\n", argv[0]);
        return 2;
    }

    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (rpc_reg == nullptr) {
        fprintf(stderr, "RPC backend is not loaded\n");
        return 1;
    }
    auto add_server = (decltype(ggml_backend_rpc_add_server) *)
            ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (add_server == nullptr) {
        fprintf(stderr, "RPC backend does not export ggml_backend_rpc_add_server\n");
        return 1;
    }

    ggml_backend_reg_t server_reg = add_server(argv[1]);
    if (server_reg == nullptr || ggml_backend_reg_dev_count(server_reg) == 0) {
        fprintf(stderr, "RPC server %s exposes no devices\n", argv[1]);
        return 1;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(server_reg, 0);

    ggml_init_params params = {
        /* .mem_size   = */ 64 * ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        fprintf(stderr, "Failed to allocate test graph context\n");
        return 1;
    }

    const std::vector<std::pair<int64_t, int64_t>> supported_shapes = {
        { 16, 256 },
        { 256, 1536 },
        { 1536, 1536 },
        { 8960, 1536 },
        { 1536, 8960 },
    };
    bool passed = true;
    for (const auto & shape : supported_shapes) {
        passed = test_shape(ctx, dev, shape.first, shape.second, true) && passed;
    }
    passed = test_shape(ctx, dev, 151936, 1536, false) && passed;

    ggml_free(ctx);
    if (!passed) {
        return 1;
    }
    printf("RPC supports_op forwarding passed for %s\n", ggml_backend_dev_name(dev));
    return 0;
}
