#include "args.h"

#include "arg.h"

void shift_args(int * argc, char ** argv, int from, int to) {
    for (int i = 0; to + i < *argc; i++) {
        argv[from + i] = argv[to + i];
    }
    *argc -= to - from;
}

void print_usage(char ** argv, moe_offload_params & params) {
    fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                           show this help message and exit\n");
    fprintf(stderr, "  -H, --host HOST                      host to bind to (default: %s)\n", params.host.c_str());
    fprintf(stderr, "  -p, --port PORT                      port to bind to (default: %d)\n", params.port);
    fprintf(stderr, "  -ms, --display-model-stats           display projected model sizes\n");
    fprintf(stderr, "  -ebv, --expert-buffer-size-vram N_GB expert buffer size in GPU memory (default: %d)\n",
            params.expert_buffer_size_vram_gb);
    fprintf(stderr, "  -ebv, --expert-buffer-size-ram N_GB  expert buffer size in system memory (default: %d)\n",
            params.expert_buffer_size_ram_gb);
    fprintf(stderr, "\n");
}

bool moe_offload_params_parse(int * argc, char ** argv, moe_offload_params & params) {
    std::string arg;
    for (int i = 1; i < *argc; i++) {
        arg = argv[i];
        if (arg == "-ebv" || arg == "--expert-buffer-size-vram") {
            if (++i >= *argc) {
                return false;
            }
            params.expert_buffer_size_vram_gb = std::stoi(argv[i]);
            shift_args(argc, argv, i - 1, i + 1);
        } else if (arg == "-ebs" || arg == "--expert-buffer-size-ram") {
            if (++i >= *argc) {
                return false;
            }
            params.expert_buffer_size_ram_gb = std::stoi(argv[i]);
            shift_args(argc, argv, i - 1, i + 1);
        } else if (arg == "-ms" || arg == "--display-model-stats") {
            params.print_model_stats = true;
            shift_args(argc, argv, i, i + 1);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv, params);
            exit(0);
        } else {
            // skip unknowns as they may be common params
        }
    }
    return common_params_parse(*argc, argv, params.common, LLAMA_EXAMPLE_SERVER);
}
