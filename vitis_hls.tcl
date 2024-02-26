open_project vitis_hls_project
set_top ViT_compute
add_files src/add.cpp
add_files src/attention.cpp
add_files src/conv.cpp
add_files src/gelu.cpp
add_files src/layernorm.cpp
add_files src/linear.cpp
add_files src/moe.cpp
add_files src/ViT_compute.cpp
add_files -tb testbench/e2e.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb weights/
open_solution "solution1" -flow_target vitis
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 300MHz -name default
config_export -format ip_catalog -rtl verilog -version 1.0.0 -vivado_clock 300MHz
config_interface -m_axi_alignment_byte_size 64 -m_axi_latency 64 -m_axi_max_widen_bitwidth 512
config_rtl -register_reset_num 3
csynth_design
