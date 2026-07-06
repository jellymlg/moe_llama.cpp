#pragma once

#include "args.h"
#include "cpp-httplib/httplib.h"
#include "model.h"

class server {
    httplib::Server    svr;
    moe_offload_params params;

    moe_model *     model;
    llama_context * ctx;

    std::mutex ctx_mutex;

    void handle_completions(const httplib::Request & req, httplib::Response & res);
    bool check_model();
  public:
    explicit server(const moe_offload_params & params);
    ~server();
    void load_model();
    void start();
};
