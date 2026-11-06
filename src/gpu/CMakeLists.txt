# Accelerated linear solvers (CFD_GPU_BACKEND: none | openmp | cuda).
add_library(cfd_gpu STATIC
    src/dispatch.cpp
    src/krylov_openmp.cpp
)
add_library(cfd::gpu ALIAS cfd_gpu)
target_include_directories(cfd_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(CFD_ENABLE_OPENMP)
    target_link_libraries(cfd_gpu PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CFD_GPU_BACKEND STREQUAL "openmp")
    if(MSVC)
        message(WARNING "CFD_GPU_BACKEND=openmp: MSVC has no OpenMP target offload; "
                        "the solver runs on host threads")
    endif()
    target_compile_definitions(cfd_gpu PRIVATE CFD_GPU_OMP_TARGET)
    # e.g. GCC: -foffload=nvptx-none, Clang: -fopenmp-targets=nvptx64 --offload-arch=sm_86
    separate_arguments(_cfd_offload_flags NATIVE_COMMAND "${CFD_OMP_OFFLOAD_FLAGS}")
    target_compile_options(cfd_gpu PRIVATE ${_cfd_offload_flags})
    if(_cfd_offload_flags)
        # The offload tables need -fopenmp on the link line (not only libgomp).
        target_link_options(cfd_gpu PUBLIC ${_cfd_offload_flags}
                            $<$<CXX_COMPILER_ID:GNU>:-fopenmp>)
    endif()
elseif(CFD_GPU_BACKEND STREQUAL "cuda")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(cfd_gpu PRIVATE src/krylov_cuda.cu)
    target_compile_definitions(cfd_gpu PUBLIC CFD_GPU_CUDA)
    target_link_libraries(cfd_gpu PUBLIC CUDA::cudart)
    set_target_properties(cfd_gpu PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
elseif(NOT CFD_GPU_BACKEND STREQUAL "none")
    message(FATAL_ERROR "CFD_GPU_BACKEND must be none, openmp or cuda (got '${CFD_GPU_BACKEND}')")
endif()

cfd_target_defaults(cfd_gpu)
