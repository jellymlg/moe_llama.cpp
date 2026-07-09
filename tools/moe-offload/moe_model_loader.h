#pragma once

#include "../../src/llama-model-loader.h"

class moe_model_loder : public llama_model_loader {
    std::vector<std::string> splits;
  public:
    moe_model_loder(const std::string & fname, const llama_model_params & params);
    bool load_all_data(ggml_context *          ctx,
                       llama_buf_map &         bufs,
                       llama_mlocks *          lmlocks,
                       llama_progress_callback progress_callback,
                       void *                  progress_callback_user_data);
};