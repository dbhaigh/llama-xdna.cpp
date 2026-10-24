#include "llama-q4nx.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-q4nx-unpack.h"

#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr uint64_t Q4NX_HEADER_MAX_SIZE = 64 * 1024 * 1024;
constexpr size_t Q4NX_ROW_BLOCK = llama_q4nx_internal::TILE_ROWS;
constexpr size_t Q4NX_COL_BLOCK = llama_q4nx_internal::TILE_COLS;
constexpr size_t Q4NX_GROUP_SIZE = 32;
constexpr size_t Q4NX_GROUPS_PER_TILE = Q4NX_COL_BLOCK / Q4NX_GROUP_SIZE;
constexpr size_t Q4NX_TILE_BYTES = 5120;
constexpr size_t Q4NX_SCALE_BYTES = Q4NX_GROUPS_PER_TILE * Q4NX_ROW_BLOCK * sizeof(uint16_t);
constexpr size_t Q4NX_MIN_BYTES = Q4NX_SCALE_BYTES;
constexpr size_t Q4NX_VALUES_OFFSET = Q4NX_SCALE_BYTES + Q4NX_MIN_BYTES;

struct q4nx_source_tensor {
    std::string gguf_name;
    std::string source_name;
    enum ggml_type type;
    uint64_t data_offset;
    uint64_t data_size;
    int64_t source_rows;
    int64_t target_rows;
    int64_t columns;
    bool matrix;
};

struct gguf_deleter {
    void operator()(gguf_context * ctx) const {
        gguf_free(ctx);
    }
};

struct ggml_deleter {
    void operator()(ggml_context * ctx) const {
        ggml_free(ctx);
    }
};

static uint64_t checked_product(uint64_t a, uint64_t b, const char * what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error(std::string("Q4NX size overflow for ") + what);
    }
    return a * b;
}

static int64_t json_i64(const json & obj, const char * key) {
    const auto it = obj.find(key);
    if (it == obj.end() || !it->is_number_integer()) {
        throw std::runtime_error(std::string("missing or invalid Q4NX config field: ") + key);
    }
    const int64_t value = it->get<int64_t>();
    if (value <= 0) {
        throw std::runtime_error(std::string("Q4NX config field must be positive: ") + key);
    }
    return value;
}

static float json_f32(const json & obj, const char * key) {
    const auto it = obj.find(key);
    if (it == obj.end() || !it->is_number()) {
        throw std::runtime_error(std::string("missing or invalid Q4NX config field: ") + key);
    }
    return it->get<float>();
}

static std::string json_string(const json & obj, const char * key) {
    const auto it = obj.find(key);
    if (it == obj.end() || !it->is_string()) {
        throw std::runtime_error(std::string("missing or invalid Q4NX config field: ") + key);
    }
    return it->get<std::string>();
}

static std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

class q4nx_model {
public:
    explicit q4nx_model(const std::string & model_path) {
        fs::path path(model_path);
        if (fs::is_directory(path)) {
            path /= "model.q4nx";
        }
        if (lowercase(path.extension().string()) != ".q4nx") {
            throw std::runtime_error("Q4NX model path must name a .q4nx file or its package directory");
        }

        model_path_ = path;
        const fs::path package_dir = path.parent_path();
        file_.open(path, std::ios::binary);
        if (!file_) {
            throw std::runtime_error("failed to open Q4NX model: " + path.string());
        }
        file_.seekg(0, std::ios::end);
        const std::streamoff file_size = file_.tellg();
        if (file_size < static_cast<std::streamoff>(sizeof(uint64_t))) {
            throw std::runtime_error("Q4NX model is truncated");
        }
        file_size_ = static_cast<uint64_t>(file_size);

        read_header();
        config_ = read_json_file(package_dir / "config.json");
        tokenizer_ = read_json_file(package_dir / "tokenizer.json");
        tokenizer_config_ = read_json_file(package_dir / "tokenizer_config.json");
        validate_config();
        build_tensor_index();
    }

    gguf_context * make_metadata() {
        std::unique_ptr<gguf_context, gguf_deleter> metadata(gguf_init_empty());
        if (!metadata) {
            throw std::runtime_error("failed to allocate GGUF metadata for Q4NX model");
        }

        gguf_set_val_str(metadata.get(), "general.architecture", "qwen2");
        gguf_set_val_str(metadata.get(), "general.name", model_path_.parent_path().filename().string().c_str());
        gguf_set_val_u32(metadata.get(), "general.alignment", 32);
        gguf_set_val_u32(metadata.get(), "general.file_type", LLAMA_FTYPE_MOSTLY_Q4_0);

        const std::string arch = "qwen2";
        gguf_set_val_u32(metadata.get(), (arch + ".context_length").c_str(), static_cast<uint32_t>(context_length_));
        gguf_set_val_u32(metadata.get(), (arch + ".embedding_length").c_str(), static_cast<uint32_t>(hidden_size_));
        gguf_set_val_u32(metadata.get(), (arch + ".block_count").c_str(), static_cast<uint32_t>(layer_count_));
        gguf_set_val_u32(metadata.get(), (arch + ".feed_forward_length").c_str(), static_cast<uint32_t>(intermediate_size_));
        gguf_set_val_u32(metadata.get(), (arch + ".attention.head_count").c_str(), static_cast<uint32_t>(head_count_));
        gguf_set_val_u32(metadata.get(), (arch + ".attention.head_count_kv").c_str(), static_cast<uint32_t>(kv_head_count_));
        gguf_set_val_u32(metadata.get(), (arch + ".attention.key_length").c_str(), static_cast<uint32_t>(head_dim_));
        gguf_set_val_u32(metadata.get(), (arch + ".attention.value_length").c_str(), static_cast<uint32_t>(head_dim_));
        gguf_set_val_f32(metadata.get(), (arch + ".attention.layer_norm_rms_epsilon").c_str(), rms_norm_eps_);
        gguf_set_val_u32(metadata.get(), (arch + ".rope.dimension_count").c_str(), static_cast<uint32_t>(head_dim_));
        gguf_set_val_f32(metadata.get(), (arch + ".rope.freq_base").c_str(), rope_theta_);

        add_tokenizer_metadata(metadata.get());
        add_tensor_metadata(metadata.get());
        return metadata.release();
    }

    void load_tensor(ggml_tensor * tensor) {
        const auto it = tensors_.find(tensor->name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Q4NX package does not contain tensor " + std::string(tensor->name));
        }

        const q4nx_source_tensor & source = it->second;
        if (tensor->type != source.type || tensor->ne[0] != source.columns ||
                tensor->ne[1] != source.target_rows || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            throw std::runtime_error("unexpected tensor shape or type while loading Q4NX tensor " + source.source_name);
        }

        if (source.type == GGML_TYPE_BF16 || source.type == GGML_TYPE_F32) {
            load_bf16_tensor(tensor, source);
        } else {
            load_q4nx_tensor(tensor, source);
        }
    }

private:
    fs::path model_path_;
    std::ifstream file_;
    uint64_t file_size_ = 0;
    uint64_t data_start_ = 0;
    json config_;
    json tokenizer_;
    json tokenizer_config_;
    std::map<std::string, q4nx_source_tensor> tensors_;
    int64_t hidden_size_ = 0;
    int64_t intermediate_size_ = 0;
    int64_t layer_count_ = 0;
    int64_t head_count_ = 0;
    int64_t kv_head_count_ = 0;
    int64_t head_dim_ = 0;
    int64_t model_vocab_size_ = 0;
    int64_t vocab_size_ = 0;
    int64_t context_length_ = 0;
    float rms_norm_eps_ = 0.0f;
    float rope_theta_ = 0.0f;

    static json read_json_file(const fs::path & path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("failed to open Q4NX companion file: " + path.string());
        }
        try {
            return json::parse(file);
        } catch (const std::exception & e) {
            throw std::runtime_error("failed to parse Q4NX companion file " + path.string() + ": " + e.what());
        }
    }

    void read_header() {
        std::array<uint8_t, sizeof(uint64_t)> length_bytes = {};
        file_.seekg(0, std::ios::beg);
        file_.read(reinterpret_cast<char *>(length_bytes.data()), length_bytes.size());
        if (!file_) {
            throw std::runtime_error("failed to read Q4NX header length");
        }

        uint64_t header_size = 0;
        for (size_t i = 0; i < length_bytes.size(); ++i) {
            header_size |= static_cast<uint64_t>(length_bytes[i]) << (8 * i);
        }
        if (header_size == 0 || header_size > Q4NX_HEADER_MAX_SIZE ||
                header_size > file_size_ - sizeof(uint64_t)) {
            throw std::runtime_error("invalid Q4NX header length");
        }

        std::string header(static_cast<size_t>(header_size), '\0');
        file_.read(header.data(), static_cast<std::streamsize>(header.size()));
        if (!file_) {
            throw std::runtime_error("failed to read Q4NX tensor header");
        }
        try {
            header_ = json::parse(header);
        } catch (const std::exception & e) {
            throw std::runtime_error(std::string("invalid Q4NX tensor header: ") + e.what());
        }
        if (!header_.is_object()) {
            throw std::runtime_error("Q4NX tensor header must be a JSON object");
        }
        data_start_ = sizeof(uint64_t) + header_size;
    }

    json header_;

    void validate_config() {
        if (json_string(config_, "model_type") != "qwen2") {
            throw std::runtime_error("native Q4NX loading currently supports Qwen2 packages only");
        }
        const auto architectures = config_.find("architectures");
        if (architectures == config_.end() || !architectures->is_array() ||
                std::find(architectures->begin(), architectures->end(), json("Qwen2ForCausalLM")) == architectures->end()) {
            throw std::runtime_error("Q4NX package is not a Qwen2ForCausalLM model");
        }

        hidden_size_ = json_i64(config_, "hidden_size");
        intermediate_size_ = json_i64(config_, "intermediate_size");
        layer_count_ = json_i64(config_, "num_hidden_layers");
        head_count_ = json_i64(config_, "num_attention_heads");
        kv_head_count_ = json_i64(config_, "num_key_value_heads");
        model_vocab_size_ = json_i64(config_, "vocab_size");
        context_length_ = json_i64(config_, "max_position_embeddings");
        head_dim_ = config_.contains("head_dim") ? json_i64(config_, "head_dim") : hidden_size_ / head_count_;
        rms_norm_eps_ = json_f32(config_, "rms_norm_eps");
        rope_theta_ = json_f32(config_, "rope_theta");

        if (hidden_size_ % head_count_ != 0 || hidden_size_ % 32 != 0 ||
                intermediate_size_ % 32 != 0 || head_dim_ <= 0 ||
                head_count_ * head_dim_ != hidden_size_ ||
                kv_head_count_ * head_dim_ % 32 != 0 ||
                model_vocab_size_ % 32 != 0) {
            throw std::runtime_error("Q4NX Qwen2 dimensions are incompatible with the supported tensor layout");
        }
        if (!tokenizer_.is_object() || !tokenizer_.contains("model") ||
                !tokenizer_["model"].is_object() || tokenizer_["model"].value("type", "") != "BPE") {
            throw std::runtime_error("Q4NX package requires a BPE tokenizer.json");
        }
        vocab_size_ = static_cast<int64_t>(tokenizer_["model"]["vocab"].size());
        if (!tokenizer_.contains("added_tokens") || !tokenizer_["added_tokens"].is_array()) {
            throw std::runtime_error("Q4NX tokenizer is missing added_tokens");
        }
        for (const auto & token : tokenizer_["added_tokens"]) {
            if (!token.contains("id") || !token["id"].is_number_integer()) {
                throw std::runtime_error("Q4NX tokenizer contains an invalid added token");
            }
            vocab_size_ = std::max(vocab_size_, token["id"].get<int64_t>() + 1);
        }
        if (vocab_size_ <= 0 || vocab_size_ > model_vocab_size_) {
            throw std::runtime_error("Q4NX tokenizer vocabulary does not fit the model embeddings");
        }
    }

    std::pair<std::string, std::string> map_tensor_name(const std::string & source_name) const {
        if (source_name == "model.embed_tokens.weight") {
            return {"token_embd.weight", source_name};
        }
        if (source_name == "model.norm.weight") {
            return {"output_norm.weight", source_name};
        }
        if (source_name == "lm_head.weight") {
            return {"output.weight", source_name};
        }
        constexpr const char * prefix = "model.layers.";
        if (source_name.compare(0, std::strlen(prefix), prefix) != 0) {
            throw std::runtime_error("unsupported tensor in Q4NX model: " + source_name);
        }

        const size_t layer_end = source_name.find('.', std::strlen(prefix));
        if (layer_end == std::string::npos) {
            throw std::runtime_error("invalid Q4NX layer tensor name: " + source_name);
        }
        const std::string layer_str = source_name.substr(std::strlen(prefix), layer_end - std::strlen(prefix));
        size_t parsed = 0;
        const int64_t layer = std::stoll(layer_str, &parsed);
        if (parsed != layer_str.size() || layer < 0 || layer >= layer_count_) {
            throw std::runtime_error("invalid Q4NX layer index in tensor: " + source_name);
        }

        const std::string suffix = source_name.substr(layer_end + 1);
        static const std::map<std::string, std::string> names = {
            {"input_layernorm.weight", "attn_norm.weight"},
            {"post_attention_layernorm.weight", "ffn_norm.weight"},
            {"self_attn.q_proj.weight", "attn_q.weight"},
            {"self_attn.k_proj.weight", "attn_k.weight"},
            {"self_attn.v_proj.weight", "attn_v.weight"},
            {"self_attn.o_proj.weight", "attn_output.weight"},
            {"self_attn.q_proj.bias", "attn_q.bias"},
            {"self_attn.k_proj.bias", "attn_k.bias"},
            {"self_attn.v_proj.bias", "attn_v.bias"},
            {"mlp.gate_proj.weight", "ffn_gate.weight"},
            {"mlp.up_proj.weight", "ffn_up.weight"},
            {"mlp.down_proj.weight", "ffn_down.weight"},
        };
        const auto it = names.find(suffix);
        if (it == names.end()) {
            throw std::runtime_error("unsupported Q4NX tensor in Qwen2 layer: " + source_name);
        }
        return {"blk." + std::to_string(layer) + "." + it->second, source_name};
    }

    q4nx_source_tensor describe_tensor(const std::string & source_name, const json & info,
            uint64_t offset, uint64_t size) const {
        const auto mapped = map_tensor_name(source_name);
        const bool is_embedding = source_name == "model.embed_tokens.weight";
        const bool is_output = source_name == "lm_head.weight";
        const bool is_weight = source_name.size() >= 7 &&
                source_name.compare(source_name.size() - 7, 7, ".weight") == 0;
        const bool is_matrix = is_embedding || is_output ||
                (is_weight && source_name.find("norm.weight") == std::string::npos);

        int64_t rows = 1;
        int64_t columns = hidden_size_;
        if (is_embedding || is_output) {
            rows = model_vocab_size_;
        } else if (source_name.find("self_attn.q_proj.weight") != std::string::npos) {
            rows = head_count_ * head_dim_;
        } else if (source_name.find("self_attn.k_proj.weight") != std::string::npos ||
                   source_name.find("self_attn.v_proj.weight") != std::string::npos) {
            rows = kv_head_count_ * head_dim_;
        } else if (source_name.find("self_attn.o_proj.weight") != std::string::npos) {
            rows = hidden_size_;
            columns = hidden_size_;
        } else if (source_name.find("mlp.gate_proj.weight") != std::string::npos ||
                   source_name.find("mlp.up_proj.weight") != std::string::npos) {
            rows = intermediate_size_;
        } else if (source_name.find("mlp.down_proj.weight") != std::string::npos) {
            rows = hidden_size_;
            columns = intermediate_size_;
        } else if (source_name.find("self_attn.q_proj.bias") != std::string::npos) {
            columns = head_count_ * head_dim_;
        } else if (source_name.find("self_attn.k_proj.bias") != std::string::npos ||
                   source_name.find("self_attn.v_proj.bias") != std::string::npos) {
            columns = kv_head_count_ * head_dim_;
        } else if (source_name.find("norm.weight") != std::string::npos ||
                   source_name == "model.norm.weight") {
            columns = hidden_size_;
        } else if (source_name.find(".bias") != std::string::npos) {
            throw std::runtime_error("unsupported Q4NX bias tensor: " + source_name);
        } else {
            throw std::runtime_error("unable to determine Q4NX tensor dimensions: " + source_name);
        }

        const int64_t target_rows = (is_embedding || is_output) ? vocab_size_ : rows;
        const bool is_bf16 = info.at("dtype") == "BF16";
        const enum ggml_type type = is_bf16 ? GGML_TYPE_F32 :
                info.at("dtype") == "I8" ? GGML_TYPE_Q4_0 :
                throw std::runtime_error("unsupported Q4NX tensor dtype for " + source_name);

        const json & shape = info.at("shape");
        if (!shape.is_array()) {
            throw std::runtime_error("invalid Q4NX tensor shape for " + source_name);
        }
        if (is_bf16) {
            if (is_matrix) {
                if (shape.size() != 2 || shape[0].get<int64_t>() != rows || shape[1].get<int64_t>() != columns) {
                    throw std::runtime_error("unexpected BF16 tensor shape in Q4NX model: " + source_name);
                }
            } else if (shape.size() != 1 || shape[0].get<int64_t>() != columns) {
                throw std::runtime_error("unexpected BF16 vector shape in Q4NX model: " + source_name);
            }
            const uint64_t expected_size = checked_product(
                    checked_product(static_cast<uint64_t>(rows), static_cast<uint64_t>(columns), source_name.c_str()),
                    sizeof(uint16_t), source_name.c_str());
            if (size != expected_size) {
                throw std::runtime_error("unexpected BF16 tensor byte size in Q4NX model: " + source_name);
            }
        } else {
            if (!is_matrix || rows % Q4NX_ROW_BLOCK != 0 || columns % Q4NX_COL_BLOCK != 0) {
                throw std::runtime_error("Q4NX matrix dimensions must be multiples of 32 rows and 256 columns: " + source_name);
            }
            const uint64_t tiles = checked_product(
                    static_cast<uint64_t>(rows / Q4NX_ROW_BLOCK),
                    static_cast<uint64_t>(columns / Q4NX_COL_BLOCK), source_name.c_str());
            if (shape.size() != 2 || shape[0].get<uint64_t>() != tiles || shape[1].get<uint64_t>() != Q4NX_TILE_BYTES ||
                    size != checked_product(tiles, Q4NX_TILE_BYTES, source_name.c_str())) {
                throw std::runtime_error("unexpected Q4NX packed matrix shape: " + source_name);
            }
        }

        return {
            mapped.first,
            mapped.second,
            type,
            offset,
            size,
            rows,
            target_rows,
            columns,
            is_matrix,
        };
    }

    void build_tensor_index() {
        if (header_.empty()) {
            throw std::runtime_error("Q4NX model contains no tensors");
        }

        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        for (auto it = header_.begin(); it != header_.end(); ++it) {
            if (it.key() == "__metadata__") {
                continue;
            }
            if (!it.value().is_object() || !it.value().contains("data_offsets")) {
                throw std::runtime_error("invalid Q4NX tensor metadata for " + it.key());
            }
            const json & data_offsets = it.value()["data_offsets"];
            if (!data_offsets.is_array() || data_offsets.size() != 2 ||
                    !data_offsets[0].is_number_unsigned() || !data_offsets[1].is_number_unsigned()) {
                throw std::runtime_error("invalid Q4NX data offsets for " + it.key());
            }
            const uint64_t begin = data_offsets[0].get<uint64_t>();
            const uint64_t end = data_offsets[1].get<uint64_t>();
            if (end < begin || end > file_size_ - data_start_) {
                throw std::runtime_error("Q4NX tensor data is outside the model file: " + it.key());
            }
            ranges.emplace_back(begin, end);

            q4nx_source_tensor tensor = describe_tensor(it.key(), it.value(), begin, end - begin);
            if (!tensors_.emplace(tensor.gguf_name, std::move(tensor)).second) {
                throw std::runtime_error("duplicate mapped Q4NX tensor name: " + it.key());
            }
        }
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].first < ranges[i - 1].second) {
                throw std::runtime_error("Q4NX tensor data ranges overlap");
            }
        }
        if (tensors_.size() != static_cast<size_t>(layer_count_) * 12 + 3) {
            throw std::runtime_error("Q4NX Qwen2 package has an unexpected tensor count");
        }
    }

    void add_tokenizer_metadata(gguf_context * metadata) const {
        const json & vocab = tokenizer_["model"]["vocab"];
        std::vector<std::string> tokens(static_cast<size_t>(vocab_size_));
        std::vector<int32_t> token_types(static_cast<size_t>(vocab_size_), LLAMA_TOKEN_TYPE_NORMAL);
        std::vector<bool> assigned(static_cast<size_t>(vocab_size_), false);

        for (auto it = vocab.begin(); it != vocab.end(); ++it) {
            if (!it.value().is_number_integer()) {
                throw std::runtime_error("Q4NX tokenizer has a non-integer token id");
            }
            const int64_t id = it.value().get<int64_t>();
            if (id < 0 || id >= vocab_size_ || assigned[id]) {
                throw std::runtime_error("Q4NX tokenizer has a duplicate or out-of-range token id");
            }
            tokens[id] = it.key();
            assigned[id] = true;
        }

        for (const auto & token : tokenizer_["added_tokens"]) {
            const int64_t id = token.at("id").get<int64_t>();
            const std::string content = json_string(token, "content");
            if (id < 0 || id >= vocab_size_ || assigned[id]) {
                throw std::runtime_error("Q4NX tokenizer has a duplicate or out-of-range added token");
            }
            tokens[id] = content;
            token_types[id] = token.value("special", false) ? LLAMA_TOKEN_TYPE_CONTROL : LLAMA_TOKEN_TYPE_USER_DEFINED;
            assigned[id] = true;
        }
        if (std::find(assigned.begin(), assigned.end(), false) != assigned.end()) {
            throw std::runtime_error("Q4NX tokenizer vocabulary contains unassigned token ids");
        }

        std::vector<const char *> token_ptrs;
        token_ptrs.reserve(tokens.size());
        for (const std::string & token : tokens) {
            token_ptrs.push_back(token.c_str());
        }
        gguf_set_val_str(metadata, "tokenizer.ggml.model", "gpt2");
        gguf_set_val_str(metadata, "tokenizer.ggml.pre", "qwen2");
        gguf_set_arr_str(metadata, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
        gguf_set_arr_data(metadata, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,
                token_types.data(), token_types.size());

        const json & merges = tokenizer_["model"]["merges"];
        if (!merges.is_array()) {
            throw std::runtime_error("Q4NX tokenizer merges must be an array");
        }
        std::vector<std::string> merge_strings;
        std::vector<const char *> merge_ptrs;
        merge_strings.reserve(merges.size());
        merge_ptrs.reserve(merges.size());
        for (const auto & merge : merges) {
            if (!merge.is_string()) {
                throw std::runtime_error("Q4NX tokenizer contains a non-string BPE merge");
            }
            merge_strings.push_back(merge.get<std::string>());
        }
        for (const std::string & merge : merge_strings) {
            merge_ptrs.push_back(merge.c_str());
        }
        gguf_set_arr_str(metadata, "tokenizer.ggml.merges", merge_ptrs.data(), merge_ptrs.size());

        if (config_.contains("bos_token_id") && config_["bos_token_id"].is_number_integer()) {
            gguf_set_val_u32(metadata, "tokenizer.ggml.bos_token_id", config_["bos_token_id"].get<uint32_t>());
        }
        if (config_.contains("eos_token_id") && config_["eos_token_id"].is_number_integer()) {
            gguf_set_val_u32(metadata, "tokenizer.ggml.eos_token_id", config_["eos_token_id"].get<uint32_t>());
        }
        const auto pad = tokenizer_config_.find("pad_token");
        if (pad != tokenizer_config_.end() && pad->is_string()) {
            const auto it = vocab.find(pad->get<std::string>());
            if (it != vocab.end() && it->is_number_integer()) {
                gguf_set_val_u32(metadata, "tokenizer.ggml.padding_token_id", it->get<uint32_t>());
            } else {
                for (const auto & added : tokenizer_["added_tokens"]) {
                    if (added.value("content", "") == pad->get<std::string>()) {
                        gguf_set_val_u32(metadata, "tokenizer.ggml.padding_token_id", added.at("id").get<uint32_t>());
                        break;
                    }
                }
            }
        }
        gguf_set_val_bool(metadata, "tokenizer.ggml.add_bos_token", tokenizer_config_.value("add_bos_token", false));
        gguf_set_val_bool(metadata, "tokenizer.ggml.add_eos_token", tokenizer_config_.value("add_eos_token", false));
        const auto chat_template = tokenizer_config_.find("chat_template");
        if (chat_template != tokenizer_config_.end() && chat_template->is_string()) {
            gguf_set_val_str(metadata, "tokenizer.chat_template", chat_template->get<std::string>().c_str());
        }
    }

    void add_tensor_metadata(gguf_context * metadata) const {
        const size_t context_size = ggml_tensor_overhead() * (tensors_.size() + 1);
        ggml_init_params params = {
            /*.mem_size   =*/ context_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        std::unique_ptr<ggml_context, ggml_deleter> tensor_ctx(ggml_init(params));
        if (!tensor_ctx) {
            throw std::runtime_error("failed to allocate Q4NX tensor metadata context");
        }

        for (const auto & item : tensors_) {
            const q4nx_source_tensor & source = item.second;
            ggml_tensor * tensor = source.matrix ?
                    ggml_new_tensor_2d(tensor_ctx.get(), source.type, source.columns, source.target_rows) :
                    ggml_new_tensor_1d(tensor_ctx.get(), source.type, source.columns);
            if (tensor == nullptr) {
                throw std::runtime_error("failed to create Q4NX tensor metadata for " + source.gguf_name);
            }
            ggml_set_name(tensor, source.gguf_name.c_str());
            gguf_add_tensor(metadata, tensor);
        }
    }

    void read_at(uint64_t offset, void * output, size_t size) {
        if (offset > file_size_ - data_start_ || size > file_size_ - data_start_ - offset) {
            throw std::runtime_error("Q4NX tensor read exceeds the model file");
        }
        file_.clear();
        file_.seekg(static_cast<std::streamoff>(data_start_ + offset), std::ios::beg);
        file_.read(static_cast<char *>(output), static_cast<std::streamsize>(size));
        if (!file_) {
            throw std::runtime_error("failed to read Q4NX tensor data");
        }
    }



    void load_bf16_tensor(ggml_tensor * tensor, const q4nx_source_tensor & source) {
        const uint64_t elements = checked_product(
                static_cast<uint64_t>(source.target_rows),
                static_cast<uint64_t>(source.columns), source.source_name.c_str());
        const size_t source_size = static_cast<size_t>(checked_product(
                elements, sizeof(uint16_t), source.source_name.c_str()));
        std::vector<uint8_t> source_data(source_size);
        read_at(source.data_offset, source_data.data(), source_data.size());

        if (tensor->type == GGML_TYPE_BF16) {
            if (ggml_nbytes(tensor) != source_size) {
                throw std::runtime_error("unexpected BF16 destination size for " + source.source_name);
            }
            ggml_backend_tensor_set(tensor, source_data.data(), 0, source_data.size());
        } else if (tensor->type == GGML_TYPE_F32) {
            std::vector<float> data(static_cast<size_t>(elements));
            if (ggml_nbytes(tensor) != data.size() * sizeof(float)) {
                throw std::runtime_error("unexpected F32 destination size for " + source.source_name);
            }
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = llama_q4nx_internal::bf16_to_f32(
                        llama_q4nx_internal::read_le_u16(source_data.data() + i * sizeof(uint16_t)));
            }
            ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
        } else {
            throw std::runtime_error("unexpected destination type for BF16 tensor " + source.source_name);
        }
    }

    void load_q4nx_tensor(ggml_tensor * tensor, const q4nx_source_tensor & source) {
        constexpr size_t q4_0_block_bytes = llama_q4nx_internal::Q4_0_BLOCK_BYTES;
        constexpr size_t q4_0_tile_row_size = Q4NX_GROUPS_PER_TILE * q4_0_block_bytes;
        const size_t row_size = ggml_row_size(GGML_TYPE_Q4_0, source.columns);
        if (ggml_nbytes(tensor) != row_size * static_cast<size_t>(source.target_rows) ||
                source.columns % Q4NX_COL_BLOCK != 0) {
            throw std::runtime_error("unexpected Q4_0 destination size for " + source.source_name);
        }

        const size_t column_tiles = static_cast<size_t>(source.columns) / Q4NX_COL_BLOCK;
        const size_t row_tiles = static_cast<size_t>(source.source_rows) / Q4NX_ROW_BLOCK;
        std::array<uint8_t, Q4NX_TILE_BYTES> tile = {};
        for (size_t tile_row = 0; tile_row < row_tiles; ++tile_row) {
            const size_t first_row = tile_row * Q4NX_ROW_BLOCK;
            if (first_row >= static_cast<size_t>(source.target_rows)) {
                break;
            }
            const size_t rows_to_write = std::min<size_t>(Q4NX_ROW_BLOCK,
                    static_cast<size_t>(source.target_rows) - first_row);
            std::vector<uint8_t> output(rows_to_write * row_size);
            for (size_t tile_col = 0; tile_col < column_tiles; ++tile_col) {
                const uint64_t tile_index = tile_row * column_tiles + tile_col;
                read_at(source.data_offset + tile_index * Q4NX_TILE_BYTES, tile.data(), tile.size());
                for (size_t row = 0; row < rows_to_write; ++row) {
                    uint8_t * output_row = output.data() + row * row_size + tile_col * q4_0_tile_row_size;
                    llama_q4nx_internal::unpack_q4_0_row(tile.data(), row, output_row);
                }
            }
            const size_t output_offset = first_row * row_size;
            ggml_backend_tensor_set(tensor, output.data(), output_offset, output.size());
        }
    }
};

static void set_q4nx_tensor_data(ggml_tensor * tensor, void * userdata) {
    static_cast<q4nx_model *>(userdata)->load_tensor(tensor);
}

} // namespace

bool llama_model_is_q4nx_path(const std::string & path) {
    const fs::path model_path(path);
    if (lowercase(model_path.extension().string()) == ".q4nx") {
        return true;
    }
    return fs::is_directory(model_path) && fs::exists(model_path / "model.q4nx");
}

llama_model * llama_model_load_q4nx_from_file(const std::string & path, llama_model_params params) {
    q4nx_model model(path);
    std::unique_ptr<gguf_context, gguf_deleter> metadata(model.make_metadata());
    return llama_model_init_from_user(metadata.get(), set_q4nx_tensor_data, &model, params);
}
