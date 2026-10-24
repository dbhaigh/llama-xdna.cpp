#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace llama_q4nx_internal {

constexpr size_t TILE_ROWS = 32;
constexpr size_t TILE_COLS = 256;
constexpr size_t GROUP_SIZE = 32;
constexpr size_t GROUPS_PER_TILE = TILE_COLS / GROUP_SIZE;
constexpr size_t Q4_0_BLOCK_BYTES = 18;
constexpr size_t SCALE_BYTES = GROUPS_PER_TILE * TILE_ROWS * sizeof(uint16_t);
constexpr size_t VALUES_OFFSET = 2 * SCALE_BYTES;
constexpr size_t PACKED_ROW_GROUP = 16;
constexpr size_t PACKED_BYTES_PER_COLUMN = PACKED_ROW_GROUP / 2;

inline uint16_t read_le_u16(const uint8_t * p) {
    return static_cast<uint16_t>(p[0]) | static_cast<uint16_t>(p[1]) << 8;
}

inline float bf16_to_f32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline void unpack_q4_0_row(const uint8_t * tile, size_t row, uint8_t * output) {
    if (row >= TILE_ROWS) {
        throw std::out_of_range("Q4NX row is outside its tile");
    }

    const size_t row_group = row / PACKED_ROW_GROUP;
    const size_t row_pair = (row % PACKED_ROW_GROUP) / 2;
    const size_t row_group_offset = row_group * TILE_COLS * PACKED_BYTES_PER_COLUMN;

    for (size_t group = 0; group < GROUPS_PER_TILE; ++group) {
        const size_t group_index = group * TILE_ROWS + row;
        const uint16_t min_bits = read_le_u16(tile + SCALE_BYTES + group_index * sizeof(uint16_t));
        if (min_bits != 0) {
            throw std::runtime_error("Q4NX loader currently supports only Q4_0-packed matrices");
        }

        const uint16_t scale = ggml_fp32_to_fp16(
                bf16_to_f32(read_le_u16(tile + group_index * sizeof(uint16_t))));
        uint8_t * block = output + group * Q4_0_BLOCK_BYTES;
        std::memcpy(block, &scale, sizeof(scale));

        for (size_t q = 0; q < GROUP_SIZE / 2; ++q) {
            const size_t col_lo = group * GROUP_SIZE + q;
            const size_t col_hi = col_lo + GROUP_SIZE / 2;
            const size_t packed_lo_index = VALUES_OFFSET + row_group_offset +
                    col_lo * PACKED_BYTES_PER_COLUMN + row_pair;
            const size_t packed_hi_index = VALUES_OFFSET + row_group_offset +
                    col_hi * PACKED_BYTES_PER_COLUMN + row_pair;
            const uint8_t packed_lo = tile[packed_lo_index];
            const uint8_t packed_hi = tile[packed_hi_index];
            const uint8_t value_lo = (row % 2 == 0 ? packed_lo & 0x0f : packed_lo >> 4) ^ 0x08;
            const uint8_t value_hi = (row % 2 == 0 ? packed_hi & 0x0f : packed_hi >> 4) ^ 0x08;
            block[2 + q] = value_lo | static_cast<uint8_t>(value_hi << 4);
        }
    }
}

}
