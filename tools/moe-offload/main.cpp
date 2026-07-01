#include "args.h"
#include "log.h"

int main(int argc, char ** argv) {
    common_init();
    llama_backend_init();
    moe_offload_params params;
    if (!moe_offload_params_parse(&argc, argv, params)) {
        return 1;
    }
    common_params_print_info(params.common);
    common_log_flush(common_log_main());
    return 0;
}
