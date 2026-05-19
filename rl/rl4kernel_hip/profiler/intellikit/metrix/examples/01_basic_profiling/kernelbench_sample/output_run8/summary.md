# KernelBench Profiling Summary

| Kernel | GPU | Compile (s) | Prewarm (s) | Inventory (s) | Filtered (s) | Post-Compile Total (s) | Serial Equivalent Total (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `10665_block_size_tuning_base` | 7 | 33.3353 | 10.4395 | 11.3849 | 55.0134 | 76.8380 | 110.1731 |
| `2405_11_convtranspose_bn_fusedtanhm_pool_groupnorm_blocksize512_base` | 6 | 33.3291 | 8.2132 | 8.7426 | 44.0481 | 61.0040 | 94.3330 |
| `2570_modular_device_functions_base_base` | 5 | 33.2256 | 7.9980 | 8.7720 | 43.6245 | 60.3947 | 93.6201 |
| `2568_unroll_critical_loops_base` | 4 | 33.1053 | 7.8856 | 8.7995 | 43.5789 | 60.2642 | 93.3693 |
| `2739_stream_leaky_relu_base` | 2 | 33.0212 | 7.9631 | 8.5672 | 43.3915 | 59.9219 | 92.9430 |
| `1795_balanced_workload_kernel_base` | 3 | 33.0334 | 7.7844 | 8.6681 | 42.8906 | 59.3432 | 92.3765 |
| `2071_optimized_shared_memory_sync_base` | 1 | 32.9729 | 7.7880 | 8.4461 | 42.8972 | 59.1315 | 92.1042 |
| `172_coalesced_tiling_kernel_base` | 0 | 7.5922 | 7.3501 | 8.3525 | 42.7391 | 58.4419 | 66.0339 |

- `Post-Compile Total (s)` = `Prewarm + Inventory + Filtered`
- `Serial Equivalent Total (s)` = `Compile + Post-Compile Total`
- `Serial Equivalent Total (s)` is a per-sample serial estimate, not the parallel end-to-end wall time.

- Effective arch: `gfx942`
- Sample count: `8`
- Metrics: `memory.hbm_bandwidth_utilization, memory.l2_hit_rate, memory.coalescing_efficiency, compute.total_flops`
- Execution mode: `parallel`
- GPU pool: `[0, 1, 2, 3, 4, 5, 6, 7]`
- Compile workers: `8`
- Prewarm iterations: `2`
- Profile iterations per run: `5`
- Forced GPU id: `None`
- Wall time (s): `117.48`

- Ignored multi-arch hints: `PYTORCH_ROCM_ARCH=gfx90a,gfx942`
- Cleared conflicting compile flags: `HIPCC_COMPILE_FLAGS_APPEND`

## Optimization opportunities

| Kernel | Priority | Optimization insight | Profiling support |
|---|---:|---|---|
| `2405_11_convtranspose_bn_fusedtanhm_pool_groupnorm_blocksize512_base` | High | The custom tanh+maxpool kernel should be rewritten around the monotonic identity `max(tanh(x)) = tanh(max(x))`, reducing 4 `tanhf` calls per output to 1. Also vectorize the 2x2 reads; the current `out_x * 2` access pattern defeats the claimed coalescing. End-to-end gains also require looking at BN/ConvTranspose, not only this kernel. | Filtered `tanh_maxpool_optimized_kernel` is `62.462 us`, with very poor coalescing `29.41%` and L2 hit rate `17.00%`. Inventory shows larger surrounding costs too: MIOpen ConvTranspose `198.283 us`, BatchNorm `228.311 us`, elementwise `62.803 us`. |
| `2568_unroll_critical_loops_base` | High | Do not optimize the unrolled ConvTranspose3D loops; remove them. Because the functional path reduces to one channel before `softmax(dim=1)`, softmax is constant `1`, so the final output is a constant `tanh(1) * scaling_factor`. A vectorized fill kernel should replace the current convolution-style computation. | Filtered kernel reports `6.030 us`, `9.8432M` FLOPs, coalescing `100%`, and HBM utilization only `1.35%`. The high FLOP count is wasted work because the final value is independent of input/weights under the profiled shape. |
| `2570_modular_device_functions_base_base` | High | Same semantic collapse as `2568`: replace the modular fused ConvTranspose3D computation with a constant fill. The device-function modularization adds structure but no useful work for this workload. | Filtered `modular_fused_operations_kernel` is `6.174 us`, `9.8432M` FLOPs, coalescing `100%`, HBM utilization `1.38%`, and L2 hit rate `50.85%`; it is slightly slower than the unrolled version while doing the same avoidable computation. |
| `1795_balanced_workload_kernel_base` | High | Replace the shared-memory tiled GEMM with a tiny-linear specialized kernel. For `128 x 10 -> 128 x 5`, `16x16` tiling wastes lanes, introduces unnecessary shared memory traffic and synchronization, and creates weak access efficiency. Compute all 5 outputs for a row in registers. | Filtered duration is `3.055 us`, but HBM utilization is only `0.0063%` and coalescing is `66.67%`. This points to launch/synchronization/lane-utilization overhead, not bandwidth saturation. |
| `2071_optimized_shared_memory_sync_base` | High | Same tiny-linear problem as `1795`. The "optimized sync" version still uses shared-memory tiles and two barriers per tile, which is unjustified when `in_features=10` and `out_features=5`. | Filtered duration is `2.990 us`, HBM utilization `0.0065%`, coalescing `66.67%`, and L2 hit rate `40.72%`. The metrics match `1795`, so the sync-focused rewrite did not address the real bottleneck. |
| `2739_stream_leaky_relu_base` | Medium | Remove multi-stream chunking. For a small contiguous LeakyReLU tensor, creating streams and launching multiple chunks is overhead; a single simple kernel, optionally vectorized with `float4`, is the right shape. | Filtered kernel duration is `2.581 us` with coalescing `100%` and HBM utilization only `0.268%`. Inventory dispatch count is `10` for 5 iterations, confirming two launches per run from the stream split. |
| `172_coalesced_tiling_kernel_base` | Medium | Keep the coalesced layout, but simplify the post-process algebra. With `scaling_factor=2.0`, `clamp(clamp(x + bias, 0, 1) * 2, 0, 1) / 2` becomes `clamp(x + bias, 0, 0.5)`. Consider `float4` vectorization and, for end-to-end speed, fusing this post-process into the ConvTranspose output path. | Filtered `post_process_coalesced` is `39.064 us`, coalescing `100%`, L2 hit rate `66.82%`, and HBM utilization `3.46%`. Inventory shows comparable surrounding costs: MIOpen ConvTranspose `34.983 us`, elementwise `42.384 us`, post-process `42.785 us`. |
| `10665_block_size_tuning_base` | Low | The custom activation kernel is already well coalesced; further block-size tuning is low leverage. Remove redundant final clamp after `tanh`, use float fast-math intrinsics where acceptable, and only pursue larger gains by fusing the activation into the GEMM epilogue. | Filtered `module_kernel_optimized` is only `2.622 us`, coalescing `100%`, L2 hit rate `49.46%`, and HBM utilization `0.117%`. Inventory shows the preceding Tensile GEMM is larger at `9.084 us`, so the custom kernel is not the primary full-path cost. |

## 10665_block_size_tuning_base

- Category: `Dataset sample`
- Note: Auto-selected dataset sample 10665_block_size_tuning_base.
- Assigned GPU: `7`
- Compile seconds: `33.3353`
- Prewarm seconds: `10.4395`
- Inventory profile seconds: `11.3849`
- Filtered profile seconds: `55.0134`
- Post-compile total seconds: `76.8380`
- Serial equivalent total seconds: `110.1731`
- HIP source: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/hip_code/10665_block_size_tuning_base.hip`
- Functional pair: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/pytorch_code_functional/10665_block_size_tuning_base.py`
- Custom kernels: `module_kernel_optimized`
- Input summary: `[{"kind": "tensor", "shape": [128, 1024], "dtype": "torch.float32", "device": "cpu"}]`
- Init summary: `[1024, 512]`
- Inventory kernels: `__amd_rocclr_fillBufferAligned, Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA2_GRVWB2_GSUAMB_GLS0_ISA942_IU1_K1_LBSPPA512_LBSPPB512_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4, module_kernel_optimized`
- Applied kernel filter: `module_kernel_optimized`
- Primary kernel: `module_kernel_optimized`
- Avg duration (us): `2.6220`
- HBM bandwidth utilization: `0.11688170116254161`
- L2 hit rate: `49.45709618525992`
- Coalescing efficiency: `100.0`
- Total FLOPs: `13110080.0`
- Interpretation: Mixed but math-heavier: arithmetic work is visible while direct HBM pressure stays comparatively low.

## 172_coalesced_tiling_kernel_base

- Category: `Conv transpose post-process`
- Note: ConvTranspose2d pipeline followed by a custom coalesced post-process kernel.
- Assigned GPU: `0`
- Compile seconds: `7.5922`
- Prewarm seconds: `7.3501`
- Inventory profile seconds: `8.3525`
- Filtered profile seconds: `42.7391`
- Post-compile total seconds: `58.4419`
- Serial equivalent total seconds: `66.0339`
- HIP source: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/hip_code/172_coalesced_tiling_kernel_base.hip`
- Functional pair: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/pytorch_code_functional/172_coalesced_tiling_kernel_base.py`
- Custom kernels: `post_process_coalesced`
- Input summary: `[{"kind": "tensor", "shape": [128, 3, 32, 32], "dtype": "torch.float32", "device": "cpu"}, 2, 1, 1, 2.0]`
- Init summary: `[3, 16, 3, 2, 1, 1, [16, 1, 1], 2.0]`
- Inventory kernels: `__amd_rocclr_fillBufferAligned, miopenSp3AsmConv_v30_3_1_gfx9_fp32_f3x2_dilation2, elementwise_kernel, post_process_coalesced`
- Applied kernel filter: `post_process_coalesced`
- Primary kernel: `post_process_coalesced`
- Avg duration (us): `39.0640`
- HBM bandwidth utilization: `3.4619069537057348`
- L2 hit rate: `66.81547619047619`
- Coalescing efficiency: `100.0`
- Total FLOPs: `676331520.0`
- Interpretation: The custom post-process is measurable, but the inventory still contains a MIOpen conv-transpose kernel; this makes the custom kernel a secondary cost in the full path.

## 1795_balanced_workload_kernel_base

- Category: `Dataset sample`
- Note: Auto-selected dataset sample 1795_balanced_workload_kernel_base.
- Assigned GPU: `3`
- Compile seconds: `33.0334`
- Prewarm seconds: `7.7844`
- Inventory profile seconds: `8.6681`
- Filtered profile seconds: `42.8906`
- Post-compile total seconds: `59.3432`
- Serial equivalent total seconds: `92.3765`
- HIP source: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/hip_code/1795_balanced_workload_kernel_base.hip`
- Functional pair: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/pytorch_code_functional/1795_balanced_workload_kernel_base.py`
- Custom kernels: `balanced_workload_kernel`
- Input summary: `[{"kind": "tensor", "shape": [128, 10], "dtype": "torch.float32", "device": "cpu"}]`
- Init summary: `[10, 5, 2.0, 1.5]`
- Inventory kernels: `balanced_workload_kernel`
- Applied kernel filter: `balanced_workload_kernel`
- Primary kernel: `balanced_workload_kernel`
- Avg duration (us): `3.0550`
- HBM bandwidth utilization: `0.00633771449135647`
- L2 hit rate: `40.723981900452486`
- Coalescing efficiency: `66.66666666666666`
- Total FLOPs: `296960.0`
- Interpretation: Memory-access inefficiency: coalescing is weaker than expected, so access pattern quality likely matters more than raw FLOPs.

## 2071_optimized_shared_memory_sync_base

- Category: `Dataset sample`
- Note: Auto-selected dataset sample 2071_optimized_shared_memory_sync_base.
- Assigned GPU: `1`
- Compile seconds: `32.9729`
- Prewarm seconds: `7.7880`
- Inventory profile seconds: `8.4461`
- Filtered profile seconds: `42.8972`
- Post-compile total seconds: `59.1315`
- Serial equivalent total seconds: `92.1042`
- HIP source: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/hip_code/2071_optimized_shared_memory_sync_base.hip`
- Functional pair: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/pytorch_code_functional/2071_optimized_shared_memory_sync_base.py`
- Custom kernels: `optimized_shared_memory_sync_kernel`
- Input summary: `[{"kind": "tensor", "shape": [128, 10], "dtype": "torch.float32", "device": "cpu"}]`
- Init summary: `[10, 5, 2.0, 1.5]`
- Inventory kernels: `optimized_shared_memory_sync_kernel`
- Applied kernel filter: `optimized_shared_memory_sync_kernel`
- Primary kernel: `optimized_shared_memory_sync_kernel`
- Avg duration (us): `2.9900`
- HBM bandwidth utilization: `0.006492449650840787`
- L2 hit rate: `40.723981900452486`
- Coalescing efficiency: `66.66666666666666`
- Total FLOPs: `296960.0`
- Interpretation: Memory-access inefficiency: coalescing is weaker than expected, so access pattern quality likely matters more than raw FLOPs.

## 2405_11_convtranspose_bn_fusedtanhm_pool_groupnorm_blocksize512_base

- Category: `Dataset sample`
- Note: Auto-selected dataset sample 2405_11_convtranspose_bn_fusedtanhm_pool_groupnorm_blocksize512_base.
- Assigned GPU: `6`
- Compile seconds: `33.3291`
- Prewarm seconds: `8.2132`
- Inventory profile seconds: `8.7426`
- Filtered profile seconds: `44.0481`
- Post-compile total seconds: `61.0040`
- Serial equivalent total seconds: `94.3330`
- HIP source: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/hip_code/2405_11_convtranspose_bn_fusedtanhm_pool_groupnorm_blocksize512_base.hip`
- Functional pair: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/pytorch_code_functional/2405_11_convtranspose_bn_fusedtanhm_pool_groupnorm_blocksize512_base.py`
- Custom kernels: `tanh_maxpool_optimized_kernel`
- Input summary: `[{"kind": "tensor", "shape": [128, 32, 32, 32], "dtype": "torch.float32", "device": "cpu"}]`
- Init summary: `[32, 64, 4, 2, 1, 8, 4]`
- Inventory kernels: `__amd_rocclr_fillBufferAligned, miopenSp3AsmConv_v30_3_1_gfx9_fp32_f3x2_dilation2, elementwise_kernel, MIOpenBatchNormFwdTrainSpatial, tanh_maxpool_optimized_kernel, RowwiseMomentsCUDAKernel, ComputeFusedParamsCUDAKernel`
- Applied kernel filter: `tanh_maxpool_optimized_kernel`
- Primary kernel: `tanh_maxpool_optimized_kernel`
- Avg duration (us): `62.4620`
- HBM bandwidth utilization: `8.108937872630849`
- L2 hit rate: `16.998188292463748`
- Coalescing efficiency: `29.411764705882355`
- Total FLOPs: `4194304000.0`
- Interpretation: Memory-access inefficiency: coalescing is weaker than expected, so access pattern quality likely matters more than raw FLOPs.

## 2568_unroll_critical_loops_base

- Category: `Dataset sample`
- Note: Auto-selected dataset sample 2568_unroll_critical_loops_base.
- Assigned GPU: `4`
- Compile seconds: `33.1053`
- Prewarm seconds: `7.8856`
- Inventory profile seconds: `8.7995`
- Filtered profile seconds: `43.5789`
- Post-compile total seconds: `60.2642`
- Serial equivalent total seconds: `93.3693`
- HIP source: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/hip_code/2568_unroll_critical_loops_base.hip`
- Functional pair: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/pytorch_code_functional/2568_unroll_critical_loops_base.py`
- Custom kernels: `unrolled_operations_kernel`
- Input summary: `[{"kind": "tensor", "shape": [16, 8, 16, 32, 32], "dtype": "torch.float32", "device": "cpu"}]`
- Init summary: `[8, 16, 3, 2, 1, [1, 1, 1, 1, 1], 2.0]`
- Inventory kernels: `unrolled_operations_kernel`
- Applied kernel filter: `unrolled_operations_kernel`
- Primary kernel: `unrolled_operations_kernel`
- Avg duration (us): `6.0300`
- HBM bandwidth utilization: `1.3470462030943553`
- L2 hit rate: `50.606680059304146`
- Coalescing efficiency: `100.0`
- Total FLOPs: `9843200.0`
- Interpretation: Mixed but math-heavier: arithmetic work is visible while direct HBM pressure stays comparatively low.

## 2570_modular_device_functions_base_base

- Category: `Dataset sample`
- Note: Auto-selected dataset sample 2570_modular_device_functions_base_base.
- Assigned GPU: `5`
- Compile seconds: `33.2256`
- Prewarm seconds: `7.9980`
- Inventory profile seconds: `8.7720`
- Filtered profile seconds: `43.6245`
- Post-compile total seconds: `60.3947`
- Serial equivalent total seconds: `93.6201`
- HIP source: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/hip_code/2570_modular_device_functions_base_base.hip`
- Functional pair: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/pytorch_code_functional/2570_modular_device_functions_base_base.py`
- Custom kernels: `modular_fused_operations_kernel`
- Input summary: `[{"kind": "tensor", "shape": [16, 8, 16, 32, 32], "dtype": "torch.float32", "device": "cpu"}]`
- Init summary: `[8, 16, 3, 2, 1, [1, 1, 1, 1, 1], 2.0]`
- Inventory kernels: `modular_fused_operations_kernel`
- Applied kernel filter: `modular_fused_operations_kernel`
- Primary kernel: `modular_fused_operations_kernel`
- Avg duration (us): `6.1740`
- HBM bandwidth utilization: `1.382081648139537`
- L2 hit rate: `50.84598303723794`
- Coalescing efficiency: `100.0`
- Total FLOPs: `9843200.0`
- Interpretation: Mixed but math-heavier: arithmetic work is visible while direct HBM pressure stays comparatively low.

## 2739_stream_leaky_relu_base

- Category: `Dataset sample`
- Note: Auto-selected dataset sample 2739_stream_leaky_relu_base.
- Assigned GPU: `2`
- Compile seconds: `33.0212`
- Prewarm seconds: `7.9631`
- Inventory profile seconds: `8.5672`
- Filtered profile seconds: `43.3915`
- Post-compile total seconds: `59.9219`
- Serial equivalent total seconds: `92.9430`
- HIP source: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/hip_code/2739_stream_leaky_relu_base.hip`
- Functional pair: `/wekafs/zepingl/rl4kernel_hip/HIP_benchmark_kit/data/hip_eval_dataset_kernelbench_25_tasks/pytorch_code_functional/2739_stream_leaky_relu_base.py`
- Custom kernels: `leaky_relu_kernel`
- Input summary: `[{"kind": "tensor", "shape": [16, 16384], "dtype": "torch.float32", "device": "cpu"}]`
- Init summary: `[0.01]`
- Inventory kernels: `leaky_relu_kernel`
- Applied kernel filter: `leaky_relu_kernel`
- Primary kernel: `leaky_relu_kernel`
- Avg duration (us): `2.5810`
- HBM bandwidth utilization: `0.26782342363113815`
- L2 hit rate: `40.25364122319129`
- Coalescing efficiency: `100.0`
- Total FLOPs: `1310720.0`
- Interpretation: Mixed but math-heavier: arithmetic work is visible while direct HBM pressure stays comparatively low.
