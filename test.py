import torch
from xformers.ops.swiglu_op import swiglu, swiglu_new


def unravel_index(flat_idx, shape):
    idxs = []
    for dim in reversed(shape):
        idxs.append(flat_idx % dim)
        flat_idx = flat_idx // dim
    return tuple(reversed(idxs))

def global_argmax(tensor):
    max_val, flat_idx = torch.max(tensor.view(-1), 0)
    max_idx = unravel_index(int(flat_idx), tensor.shape)
    return max_idx, max_val.item()
  
def p1_error(y: torch.Tensor, z: torch.Tensor) -> float:
    '''Calculate the P1-error between two tensors. y is the ground truth.'''
    element_wise_error = torch.abs(z - y) / (torch.abs(y) + 1)
    index, value = global_argmax(element_wise_error)
    return index, value
    


x = torch.randn(64, 4096, device=torch.device("cuda"), dtype=torch.float16)
gate = torch.nn.Linear(4096, 11008, bias=False, device=torch.device("cuda"), dtype=torch.float16)
up = torch.nn.Linear(4096, 11008, bias=False, device=torch.device("cuda"), dtype=torch.float16)
down = torch.nn.Linear(11008, 4096, bias=False, device=torch.device("cuda"), dtype=torch.float16)
act = torch.nn.SiLU()

output = torch.ops.xformers.dual_gemm_silu_identity_mul(x, gate.weight, None, up.weight, None)
gx = gate(x)
ux = up(x)
print (f"fused output = {output}")
print (f"gx = {gx}")
print (f"ux = {ux}")
print (f"P1 Error of gx: {p1_error(gx, output[0])}")
print (f"P1 Error of ux: {p1_error(ux, output[1])}")

z = torch.ops.xformers.silu_identity_mul_gemm(gx, ux, down.weight, None)
swiglu_z = swiglu(x, gate.weight, None, up.weight, None, down.weight, None, op=None)
swiglu_new_z = swiglu_new(x, gate.weight, None, up.weight, None, down.weight, None, op=None)
y = down(act(gx) * ux)
print (z)
print (swiglu_z)
print (swiglu_new_z)
print (y)
print (f"P1 Error of z: {p1_error(y, z)}")
print (f"P1 Error of swiglu_z: {p1_error(y, swiglu_z)}")
print (f"P1 Error of swiglu_new_z: {p1_error(y, swiglu_new_z)}")


