#include "common.h"
#include "log.h"

int main(int argc, char ** argv) {
    common_init();
    llama_backend_init();
    common_log_add(common_log_main(), GGML_LOG_LEVEL_INFO, "Hello World");
    common_log_flush(common_log_main());
    return 0;
}