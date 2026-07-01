set(TARGET moe-offload)
add_executable(${TARGET}
        main.cpp
        args.cpp
        args.h
        server.cpp
        server.h)
target_link_libraries(${TARGET} PRIVATE ggml llama-common)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif ()