#pragma once

#include "common.h"

#include <string>

struct moe_offload_params {
    std::string   host                       = "127.0.0.1";
    int           port                       = 22333;
    int           expert_buffer_size_vram_gb = 4;
    int           expert_buffer_size_ram_gb  = 4;
    bool          print_model_stats          = false;
    common_params common;
};

void shift_args(int * argc, char ** argv, int from, int to);

void print_usage(char ** argv, moe_offload_params & params);

bool moe_offload_params_parse(int * argc, char ** argv, moe_offload_params & params);
