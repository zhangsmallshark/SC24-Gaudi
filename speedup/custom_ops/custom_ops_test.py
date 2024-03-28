import torch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht
import time
import os
import torch.nn.functional as F

import sys
# heads = int(sys.argv[1])
# print(f'num heads: {heads}')

# batches = int(sys.argv[1])
# print(f'num batches: {batches}')

torch.manual_seed(6)
print(torch.__version__)

# export GC_KERNEL_PATH=/home/chengming.zhang/exps/fc_nn/custom_mm1/Habana_Custom_Kernel/build/src/libcustom_tpc_perf_lib.so:/usr/lib/habanalabs/libtpc_kernels.so
custom_ops_op_lib_path = "/home/chengming.zhang/exps/fc_nn/custom_ops/build/lib.linux-x86_64-cpython-310/hpu_custom_ops.cpython-310-x86_64-linux-gnu.so"
torch.ops.load_library(custom_ops_op_lib_path)

device = torch.device("hpu")
torch.cuda.current_device = lambda: None
torch.cuda.set_device = lambda x: None

lazy_mode = True
if lazy_mode:
    os.environ["PT_HPU_LAZY_MODE"] = "1"
else:
    os.environ["PT_HPU_LAZY_MODE"] = "2"

# os.environ['LOG_LEVEL_ALL'] = str(1)
# os.environ['TPC_RUNNER'] = str(1)
# os.environ['HABANA_PROFILE'] = str(1)

hidden = 64
rows = 1024
heads = 8
batches = 16

in0 = torch.randn(rows*batches, 128, dtype=torch.float32, requires_grad=False)
in0_d = in0.to(device)
# out0 = torch.ops.custom_op.custom_softmax(in0_d, 0)

in0 = torch.randn(batches, heads, rows, hidden, dtype=torch.float32, requires_grad=False)
in1 = torch.randn(batches, heads, rows, hidden, dtype=torch.float32, requires_grad=False)
in2 = torch.randn(batches, heads, rows, hidden, dtype=torch.float32, requires_grad=False)
in3 = torch.randn(batches, heads, rows, hidden, dtype=torch.float32, requires_grad=False)

in0_d = in0.to(device)
in1_d = in1.to(device)
in2_d = in2.to(device)
in3_d = in3.to(device)

def test_custom_outer_op_function(in0, in1):
    out0 = torch.ops.custom_op.custom_outer(in0, in1)
    return out0

def test_torch(in0, in1):
    out1 = in0.unsqueeze(4) @ in1.unsqueeze(3)
    return out1


# out0 = test_custom_outer_op_function(in0_d, in1_d)
# print(out0.shape)
# out0=out0.reshape(batches, heads, rows, hidden, hidden)
# # out1 = test_torch(in2_d, in3_d)
# out1 = test_torch(in0_d, in1_d)
# print('custom out: ')
# print(out0)
# print('\nref out: ')
# print(out1)

out0 = torch.ops.custom_op.custom_outer(in0_d, in1_d) #TPC
# out1 = in2_d.unsqueeze(4) @ in3_d.unsqueeze(3) #MME

# @torch.no_grad()
# def overlap_custom_operation(in0, in1, in2, in3):
#     out0 = torch.ops.custom_op.custom_outer(in0, in1) #TPC
#     out1 = in2.unsqueeze(4) @ in3.unsqueeze(3) #MME
#     return out0, out1

# out0, out1 = overlap_custom_operation(in0_d, in1_d, in2_d, in3_d)
# out0, out1 = overlap_custom_operation(in0_d, in1_d, in0_d, in1_d)

#export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=7516192768
#export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4

htcore.mark_step()
print(out0)

# ht.hpu.synchronize()

print('finish !')