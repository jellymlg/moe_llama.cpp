#include "server.h"

#include "../../src/llama-model.h"
#include "log.h"

void server::handle_completions(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard lock(ctx_mutex);
    // TODO
}

bool server::check_model() {
    params.common.no_alloc          = true;
    const llama_model_params mp     = common_model_params_to_llama(params.common);
    llama_model *            m      = llama_model_load_from_file(params.common.model.path.c_str(), mp);
    const bool               is_moe = m->hparams.n_expert > 0 && m->hparams.n_expert_used > 0;
    params.common.no_alloc          = false;
    llama_model_free(m);
    return is_moe;
}

server::server(const moe_offload_params & params) : params(params) {
    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
    svr.Post("/v1/chat/completions",
             static_cast<httplib::Server::Handler>(
                 std::bind(&server::handle_completions, this, std::placeholders::_1, std::placeholders::_2)));
}

server::~server() {
    svr.stop();
    llama_free(ctx);
    llama_model_free(model);
}

void server::load_model() {
    //if (!check_model()) {
    //    common_log_add(common_log_main(), GGML_LOG_LEVEL_ERROR, "The loaded model is not MoE");
    //    exit(1);
    //}
    //params.common.no_alloc        = true;
    const llama_model_params mp   = common_model_params_to_llama(params.common);
    model                         = create_moe_model(params.common.model.path.c_str(), mp);
    const llama_context_params cp = common_context_params_to_llama(params.common);
    //ctx                         = llama_init_from_model(model, cp);
}

void server::start() {
    svr.listen(params.host, params.port);
}
