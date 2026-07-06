#pragma once

#include "../../src/llama-model-loader.h"
#include "../../src/llama-model.h"

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

class moe_model : public llama_model_base {
  public:
    explicit moe_model(const llama_model_params & params);
    bool load_tensors(moe_model_loder & ml);
};

moe_model * create_moe_model(const std::string & file, const llama_model_params & params);
