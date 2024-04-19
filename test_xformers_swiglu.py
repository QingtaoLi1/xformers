import os
import time
import torch
import torch.nn as nn
from xformers.ops.swiglu_op import swiglu, swiglu_new, swiglu_packed, swiglu_packed_new, SwiGLUPackedFusedOp, SwiGLUPackedFusedOpNew
from flash_attn.ops.activations import swiglu as swiglu_flashattn


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
    

def test_forward_time(repeat, module, *args) -> float:
    warmup = 100
    for i in range(warmup):
        y = module(*args)

    elapsed_time = 0
    for i in range(repeat):
        # Re-random the input tensor(s)
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dtype == torch.float16 and arg.requires_grad:
                arg.requires_grad = False
                arg.normal_()
                arg.requires_grad = True

        start = time.time()
        y = module(*args)
        end = time.time()
        elapsed_time += (end-start)
        del y
    print (f"{module} forward time: {elapsed_time/repeat*1000} ms.")
    return elapsed_time/repeat*1000

def test_backward_time(repeat, module, *args) -> float:
    warmup = 100
    for i in range(warmup):
        y = module(*args)
        if isinstance(y, tuple):
            loss = sum([yi.sum() for yi in y if isinstance(yi, torch.Tensor)])
        else:
            loss = y.sum()
        loss.backward()
        
    elapsed_time = 0
    for i in range(repeat):
        # module.zero_grad()
        # Re-random the input tensor(s)
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dtype == torch.float16 and arg.requires_grad:
                arg.requires_grad = False
                arg.normal_()
                arg.requires_grad = True

        y = module(*args)
        if isinstance(y, tuple):
            loss = sum([yi.sum() for yi in y if isinstance(yi, torch.Tensor)])
        else:
            loss = y.sum()
        start = time.time()
        loss.backward()
        end = time.time()
        elapsed_time += (end-start)
        del loss, y
    print (f"{module} backward time: {elapsed_time/repeat*1000} ms.")
    return elapsed_time/repeat*1000

def check_p1_error(func_name, ref, fused) -> float:
    index, err_value = p1_error(ref, fused)
    print (f"P1-error of {func_name:<20}: {err_value:<10.5f}, ref = {ref[index]:<10.5f}, fused = {fused[index]:<10.5f}\t at {index}")
    return err_value


class SwigluOriginUnpacked(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class SwigluOriginPacked(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        mma_intermediate_states = self.gate_proj(x).chunk(2, dim=-1)
        gx = mma_intermediate_states[0]
        ux = mma_intermediate_states[1]
        down_proj = self.down_proj(self.act_fn(gx) * ux)
        return down_proj


class SwigluFlashattnUnpacked(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        print (f"Max reserved memory before gx  : {torch.cuda.max_memory_reserved() / 1024 / 1024} MB")
        print (f"Max allocated memory before gx : {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        gx = self.gate_proj(x)
        print (f"Max reserved memory after gx   : {torch.cuda.max_memory_reserved() / 1024 / 1024} MB")
        print (f"Max allocated memory after gx  : {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        ux = self.up_proj(x)
        print (f"Max reserved memory after ux   : {torch.cuda.max_memory_reserved() / 1024 / 1024} MB")
        print (f"Max allocated memory after ux  : {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        intermediate_states = swiglu_flashattn(gx, ux)
        print (f"Max reserved memory after swi  : {torch.cuda.max_memory_reserved() / 1024 / 1024} MB")
        print (f"Max allocated memory after swi : {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        down_proj = self.down_proj(intermediate_states)
        print (f"Max reserved memory after down : {torch.cuda.max_memory_reserved() / 1024 / 1024} MB")
        print (f"Max allocated memory after down: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        return down_proj

class SwigluFlashattnPacked(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        mma_intermediate_states = self.gate_proj(x).chunk(2, dim=-1)
        gx = mma_intermediate_states[0]
        ux = mma_intermediate_states[1]
        intermediate_states = swiglu_flashattn(gx, ux)
        down_proj = self.down_proj(intermediate_states)
        return down_proj


class SwigluXformersUnpacked(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = swiglu(x, self.gate_proj.weight, None, self.up_proj.weight, None, self.down_proj.weight, None)
        return down_proj

class SwigluXformersPacked(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        w1w2 = self.gate_proj.weight.view([2, self.gate_proj.weight.shape[0] // 2, self.gate_proj.weight.shape[1]])
        down_proj = swiglu_packed(x, w1w2, None, self.down_proj.weight, None, op=SwiGLUPackedFusedOp)
        return down_proj


class SwigluXformersNewUnpacked(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = swiglu_new(x, self.gate_proj.weight, None, self.up_proj.weight, None, self.down_proj.weight, None)
        return down_proj

class SwigluXformersNewPacked(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        w1w2 = self.gate_proj.weight.view([2, self.gate_proj.weight.shape[0] // 2, self.gate_proj.weight.shape[1]])
        down_proj = swiglu_packed_new(x, w1w2, None, self.down_proj.weight, None, op=SwiGLUPackedFusedOpNew)
        return down_proj



if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_dtype(torch.float16)

    arches = ['A100']
    batch_size = 1
    seq_lens = [1024]
    hidden_sizes = [4096]
    intermediate_sizes = [11008]

    for arch in arches:
        os.environ["WELDER_ARCH"] = arch
        
        for hidden_size, intermediate_size in zip(hidden_sizes, intermediate_sizes):
            for seq_len in seq_lens:
                x = torch.randn(batch_size, seq_len, hidden_size, requires_grad = True, device=device)
                # module = SwigluOriginUnpacked(hidden_size, intermediate_size).to(device)
                # module = SwigluOriginPacked(hidden_size, intermediate_size).to(device)
                # module = SwigluFlashattnUnpacked(hidden_size, intermediate_size).to(device)
                # module = SwigluFlashattnPacked(hidden_size, intermediate_size).to(device)
                # module = SwigluXformersUnpacked(hidden_size, intermediate_size).to(device)
                # module = SwigluXformersPacked(hidden_size, intermediate_size).to(device)
                module = SwigluXformersNewUnpacked(hidden_size, intermediate_size).to(device)
                # module = SwigluXformersNewPacked(hidden_size, intermediate_size).to(device)

                # y_ref0 = module(x)
                # loss_ref0 = y_ref0.sum()
                # loss_ref0.backward()

                torch.cuda.memory._record_memory_history()

                # Check efficiency
                print ("------ Efficiency Check ------")
                repeat = 1
                print (f"Max reserved memory before forward  : {torch.cuda.max_memory_reserved() / 1024 / 1024} MB")
                print (f"Max allocated memory before forward : {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
                # time_forward_ref = test_forward_time(repeat, module, x)
                y = module(x)
                print (f"Max reserved memory after forward   : {torch.cuda.max_memory_reserved() / 1024 / 1024} MB")
                print (f"Max allocated memory after forward  : {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
                # time_backward_ref = test_backward_time(repeat, module, x)
                loss = y.sum()
                loss.backward()
                print (f"Max reserved memory after backward  : {torch.cuda.max_memory_reserved() / 1024 / 1024} MB")
                print (f"Max allocated memory after backward : {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")

                # torch.cuda.memory._dump_snapshot("snapshot_xnew_improved_0416.pickle")
                del x, module
