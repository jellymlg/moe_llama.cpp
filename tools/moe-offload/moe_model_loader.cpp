#include "moe_model_loader.h"
#include "log.h"

moe_model_loder::moe_model_loder(const std::string & fname, const llama_model_params & params) :
    llama_model_loader(nullptr,
                       nullptr,
                       nullptr,
                       fname,
                       splits,
                       nullptr,
                       params.use_mmap,
                       params.use_direct_io,
                       params.check_tensors,
                       params.no_alloc,
                       params.kv_overrides,
                       params.tensor_buft_overrides) {}

bool moe_model_loder::load_all_data(ggml_context *          ctx,
                                    llama_buf_map &         bufs,
                                    llama_mlocks *          lmlocks,
                                    llama_progress_callback progress_callback,
                                    void *                  progress_callback_user_data) {
    common_log_add(common_log_main(), GGML_LOG_LEVEL_INFO, "moe_model_loder::load_all_data\n");
    return true;
}