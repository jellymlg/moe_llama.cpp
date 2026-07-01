#include "server.h"

void server::handle_completions(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard lock(ctx_mutex);
    // TODO
}

server::server(const moe_offload_params & params) : params(params) {
    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
    svr.Post("/v1/chat/completions", std::bind_front(&server::handle_completions, this));
}

server::~server() {
    svr.stop();
    llama_free(ctx);
    llama_model_free(model);
}

void server::load_model() {
    const llama_model_params mp   = common_model_params_to_llama(params.common);
    model                         = llama_model_load_from_file(params.common.model.path.c_str(), mp);
    const llama_context_params cp = common_context_params_to_llama(params.common);
    ctx                           = llama_init_from_model(model, cp);
}

void server::start() {
    svr.listen(params.host, params.port);
}
