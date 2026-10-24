#define NOCPP

#include <stddef.h>
#include <stdint.h>

static float fp16_to_fp32(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                shift++;
            }
            mantissa &= 0x03ffu;
            exponent = static_cast<uint32_t>(127 - 14 - shift);
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        exponent = exponent - 15 + 127;
        bits = sign | (exponent << 23) | (mantissa << 13);
    }

    union {
        uint32_t u;
        float f;
    } result = { bits };
    return result.f;
}

extern "C" {
void q4_0_matvec(const uint8_t *restrict weights, const float *restrict input, float *restrict output, int32_t k_tile) {
    const int input_offset = k_tile * 256;
    for (int row = 0; row < 16; row++) {
        float sum = k_tile == 0 ? 0.0f : output[row];
        for (int block = 0; block < 8; block++) {
            const size_t block_index = static_cast<size_t>(row * 8 + block);
            const uint8_t * block_data = weights + block_index * 18;
            const uint16_t scale_bits = static_cast<uint16_t>(block_data[0]) |
                    (static_cast<uint16_t>(block_data[1]) << 8);
            const float scale = fp16_to_fp32(scale_bits);
            const uint8_t * quants = block_data + 2;
            const int block_input_offset = input_offset + block * 32;
            float block_sum = 0.0f;

            for (int i = 0; i < 16; i++) {
                const int q0 = static_cast<int>(quants[i] & 0x0fu) - 8;
                const int q1 = static_cast<int>(quants[i] >> 4) - 8;
                block_sum += static_cast<float>(q0) * input[block_input_offset + i];
                block_sum += static_cast<float>(q1) * input[block_input_offset + i + 16];
            }
            sum += scale * block_sum;
        }
        output[row] = sum;
    }
}
}
