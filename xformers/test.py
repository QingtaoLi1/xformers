# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import swiglu_extension

print (dir(swiglu_extension))
print (swiglu_extension.dual_gemm_silu_identity_mul)
print (swiglu_extension.swiglu_packedw)
print (torch.ops.swiglu_extension.dual_gemm_silu_identity_mul)
print (torch.ops.swiglu_extension.swiglu_packedw)


