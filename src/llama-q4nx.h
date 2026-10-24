#pragma once

#include "llama.h"

#include <string>

bool llama_model_is_q4nx_path(const std::string & path);
llama_model * llama_model_load_q4nx_from_file(const std::string & path, llama_model_params params);
