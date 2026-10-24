#include "llama-q4nx-unpack.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

static uint16_t float_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

static uint8_t code_for(size_t row, size_t col) {
    return static_cast<uint8_t>((row * 7 + col * 3) & 0x0f);
}

int main() {
    std::array<uint8_t, 5120> tile = {};
    for (size_t group = 0; group < llama_q4nx_internal::GROUPS_PER_TILE; ++group) {
        for (size_t row = 0; row < llama_q4nx_internal::TILE_ROWS; ++row) {
            const size_t scale_index = group * llama_q4nx_internal::TILE_ROWS + row;
            const uint16_t scale = float_to_bf16(1.0f + group * 0.25f + row * 0.015625f);
            const size_t offset = scale_index * sizeof(uint16_t);
            tile[offset] = static_cast<uint8_t>(scale);
            tile[offset + 1] = static_cast<uint8_t>(scale >> 8);
        }
    }

    for (size_t row = 0; row < llama_q4nx_internal::TILE_ROWS; ++row) {
        for (size_t col = 0; col < llama_q4nx_internal::TILE_COLS; ++col) {
            const size_t offset = llama_q4nx_internal::VALUES_OFFSET +
                    (row / llama_q4nx_internal::PACKED_ROW_GROUP) *
                    llama_q4nx_internal::TILE_COLS * llama_q4nx_internal::PACKED_BYTES_PER_COLUMN +
                    col * llama_q4nx_internal::PACKED_BYTES_PER_COLUMN +
                    (row % llama_q4nx_internal::PACKED_ROW_GROUP) / 2;
            const uint8_t code = code_for(row, col);
            if (row % 2 == 0) {
                tile[offset] = (tile[offset] & 0xf0) | code;
            } else {
                tile[offset] = (tile[offset] & 0x0f) | static_cast<uint8_t>(code << 4);
            }
        }
    }

    std::array<uint8_t, llama_q4nx_internal::GROUPS_PER_TILE *
            llama_q4nx_internal::Q4_0_BLOCK_BYTES> output = {};
    for (size_t row = 0; row < llama_q4nx_internal::TILE_ROWS; ++row) {
        llama_q4nx_internal::unpack_q4_0_row(tile.data(), row, output.data());
        for (size_t group = 0; group < llama_q4nx_internal::GROUPS_PER_TILE; ++group) {
            const size_t block_offset = group * llama_q4nx_internal::Q4_0_BLOCK_BYTES;
            const size_t scale_index = group * llama_q4nx_internal::TILE_ROWS + row;
            const uint16_t scale_bits = static_cast<uint16_t>(tile[scale_index * 2]) |
                    static_cast<uint16_t>(tile[scale_index * 2 + 1]) << 8;
            const uint16_t expected_scale = ggml_fp32_to_fp16(
                    llama_q4nx_internal::bf16_to_f32(scale_bits));
            uint16_t actual_scale;
            std::memcpy(&actual_scale, output.data() + block_offset, sizeof(actual_scale));
            if (actual_scale != expected_scale) {
                std::fprintf(stderr, "wrong scale at row %zu group %zu\n", row, group);
                return 1;
            }

            for (size_t q = 0; q < llama_q4nx_internal::GROUP_SIZE / 2; ++q) {
                const size_t col_lo = group * llama_q4nx_internal::GROUP_SIZE + q;
                const size_t col_hi = col_lo + llama_q4nx_internal::GROUP_SIZE / 2;
                const uint8_t value_lo = code_for(row, col_lo) ^ 0x08;
                const uint8_t value_hi = code_for(row, col_hi) ^ 0x08;
                const uint8_t expected = value_lo | static_cast<uint8_t>(value_hi << 4);
                if (output[block_offset + 2 + q] != expected) {
                    std::fprintf(stderr, "wrong quant at row %zu group %zu block offset %zu\n", row, group, q);
                    return 1;
                }
            }
        }
    }

    tile[llama_q4nx_internal::SCALE_BYTES] = 1;
    try {
        llama_q4nx_internal::unpack_q4_0_row(tile.data(), 0, output.data());
        std::fprintf(stderr, "non-zero Q4NX minima should be rejected\n");
        return 1;
    } catch (const std::runtime_error &) {
    }

    return 0;
}
