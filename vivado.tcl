create_project Edge-MoE vivado_project -part xczu9eg-ffvb1156-2-e
set_property board_part xilinx.com:zcu102:part0:3.4 [current_project]
set_property  ip_repo_paths  vitis_hls_project/solution1/impl [current_project]
update_ip_catalog
create_bd_design "design_1"
update_compile_order -fileset sources_1
startgroup
create_bd_cell -type ip -vlnv xilinx.com:hls:ViT_compute:1.0 ViT_compute_0
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 zynq_ultra_ps_e_0
endgroup
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list CONFIG.PSU__USE__S_AXI_GP0 {1} CONFIG.PSU__USE__S_AXI_GP1 {1} CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__USE__S_AXI_GP3 {1} CONFIG.PSU__USE__S_AXI_GP4 {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {300}] [get_bd_cells zynq_ultra_ps_e_0]
startgroup
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/ViT_compute_0/s_axi_control} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins ViT_compute_0/s_axi_control]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/ViT_compute_0/m_axi_inout2} Slave {/zynq_ultra_ps_e_0/S_AXI_HP0_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/ViT_compute_0/m_axi_inout3} Slave {/zynq_ultra_ps_e_0/S_AXI_HP1_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP1_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/ViT_compute_0/m_axi_inout4} Slave {/zynq_ultra_ps_e_0/S_AXI_HP2_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP2_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/ViT_compute_0/m_axi_weights} Slave {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/ViT_compute_0/m_axi_inout1} Slave {/zynq_ultra_ps_e_0/S_AXI_HPC1_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC1_FPD]
endgroup
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {/zynq_ultra_ps_e_0/pl_clk0 (299 MHz)} Clk_xbar {/zynq_ultra_ps_e_0/pl_clk0 (299 MHz)} Master {/zynq_ultra_ps_e_0/M_AXI_HPM1_FPD} Slave {/ViT_compute_0/s_axi_control} ddr_seg {Auto} intc_ip {/ps8_0_axi_periph} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM1_FPD]
save_bd_design
make_wrapper -files [get_files vivado_project/Edge-MoE.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse vivado_project/Edge-MoE.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v
set_property strategy Performance_Auto_1 [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs 32
