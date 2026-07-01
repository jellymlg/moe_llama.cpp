#pragma once

#include "args.h"
#include "cpp-httplib/httplib.h"

class server {
    httplib::Server    svr;
    moe_offload_params params;

    llama_model *   model;
    llama_context * ctx;

    std::mutex ctx_mutex;

    void handle_completions(const httplib::Request & req, httplib::Response & res);
  public:
    explicit server(const moe_offload_params & params);
    ~server();
    void load_model();
    void start();
};
