#include "args.h"
#include "log.h"
#include "server.h"

int main(int argc, char ** argv) {
    common_init();
    llama_backend_init();
    moe_offload_params params;
    if (!moe_offload_params_parse(&argc, argv, params)) {
        return 1;
    }
    common_params_print_info(params.common, true);
    common_log_flush(common_log_main());
    server server(params);
    server.load_model();
    server.start();
    return 0;
}
