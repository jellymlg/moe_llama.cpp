#include "model.h"

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

moe_model::moe_model(const llama_model_params & params) : llama_model_base(params) {}

bool moe_model::load_tensors(moe_model_loder & ml) {
    common_log_add(common_log_main(), GGML_LOG_LEVEL_INFO, "moe_model::load_tensors\n");
    ggml_context * ctx;
    llama_buf_map  buf_map;
    llama_mlocks   mlock_mmaps;
    ml.load_all_data(ctx, buf_map, params.use_mlock ? &mlock_mmaps : NULL, params.progress_callback,
                     params.progress_callback_user_data);
    return true;
}

moe_model * create_moe_model(const std::string & file, const llama_model_params & params) {
    moe_model_loder ml(file, params);
    //ml.print_info();
    auto            model = static_cast<moe_model *>(llama_model_create(ml, params));
    if (model == nullptr) {
        GGML_ABORT("fatal error: model cannot be cast to moe_model");
    }
    model->t_load_us = 0;
    time_meas tm(model->t_load_us);
    model->t_start_us = tm.t_start_us;
    model->load_hparams(ml);
    model->load_vocab(ml);
    model->load_stats(ml);
    //model->print_info();
    auto t = model->type_name();
    auto a = model->arch_name();
    common_log_add(common_log_main(), GGML_LOG_LEVEL_INFO, "type: %s, arch %s\n", t.c_str(), a.c_str());
    return model->load_tensors(ml) ? model : nullptr;
}
