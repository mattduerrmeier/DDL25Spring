from simplellm.llama import (
    LLamaFirstStage,
    LLamaLastStage,
    LLamaStage,
)  # get our models
from simplellm.tokenizers import SPTokenizer  # get our tokenizer
from simplellm.dataloaders import TinyStories  # get our dataset
from simplellm.losses import causalLLMLoss  # our loss
from torch.optim import Adam
import torch.distributed as dist
import torch
import os
from sys import argv
# contact: abele.malan@unine.ch

rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
world_size = 3 * 2
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)

# two pipelines
# 1. fwd, bwd, wait
# P1: 0 -> 1 -> 2
# P2: 3 -> 4 -> 5
# 2. exchange gradients:
# 0 <-> 3
# 1 <-> 4
# 2 <-> 5
# finally, update weights

group_0 = dist.new_group([0, 3], backend="gloo")
group_1 = dist.new_group([1, 4], backend="gloo")
group_2 = dist.new_group([2, 5], backend="gloo")

torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6 // world_size
seq_l = 256
batch_size = 3
device = "cuda"

# make the model
if rank == 0 or rank == 3:
    tokenizer = SPTokenizer()
    net = LLamaFirstStage(
        tokenizer.vocab_size,
        dmodel=dmodel,
        num_heads=num_heads,
        device=device,
        n_layers=n_layers,
        ctx_size=seq_l,
    )
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)  # no skip
    iter_ds = iter(ds)

elif rank == 1 or rank == 4:
    net = LLamaStage(
        dmodel=dmodel,
        num_heads=num_heads,
        device=device,
        n_layers=n_layers,
        ctx_size=seq_l,
    )

elif rank == 2 or rank == 5:
    tokenizer = SPTokenizer()
    net = LLamaLastStage(
        tokenizer.vocab_size,
        dmodel=dmodel,
        num_heads=num_heads,
        device=device,
        n_layers=n_layers,
        ctx_size=seq_l,
    )
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)  # no skip
    iter_ds = iter(ds)


optim = Adam(net.parameters(), lr=8e-4)


sizes = []
len_sizes = []
for param in net.parameters():
    sizes.append(param.shape)
    len_sizes.append(len(param.view(-1)))

n_micro_batch = 3
micro_batch_size = batch_size // n_micro_batch

for outer_iter in range(5_000):
    # FORWARD PASS:
    optim.zero_grad()
    reqs_wait = []
    if rank == 0 or rank == 3:
        # cut the mini batch into micro-batches
        mini_batch = next(iter_ds)
        micro_batches = torch.chunk(mini_batch, n_micro_batch)

        # accumulate the activations
        acc_outs = []

        for iter_micro_batch, out in enumerate(micro_batches):
            tag = iter_micro_batch + outer_iter

            out = out.to(device)
            out = net.embed(out)
            acc_outs.append(out)

            req = dist.isend(out.to("cpu"), dst=rank + 1, tag=tag)
            reqs_wait.append(req)

    elif rank == 1 or rank == 4:
        # accumulate the inputs and activations
        acc_outs = []
        acc_input_batches = []

        for iter_micro_batch in range(n_micro_batch):
            tag = iter_micro_batch + outer_iter
            inp_batch = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(inp_batch, src=rank - 1, tag=tag)
            req.wait()

            with torch.no_grad():
                inp_batch = inp_batch.to(device)
                inp_batch.requires_grad_()
                inp_batch.retain_grad()

            out = net(inp_batch)
            acc_outs.append(out)
            acc_input_batches.append(inp_batch)

            req = dist.isend(out.to("cpu"), rank + 1, tag=tag)
            reqs_wait.append(req)

    elif rank == 2 or rank == 5:
        # accumulate the inputs
        acc_input_batches = []

        target_mini_batch = next(iter_ds)
        target_micro_batches = torch.chunk(target_mini_batch, n_micro_batch)

        for iter_micro_batch, target in enumerate(target_micro_batches):
            tag = iter_micro_batch + outer_iter
            inp_batch = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(inp_batch, src=rank - 1, tag=tag)
            req.wait()

            with torch.no_grad():
                inp_batch = inp_batch.to(device)
                inp_batch.requires_grad_()
                inp_batch.retain_grad()

            logits = net(inp_batch)
            loss = causalLLMLoss(logits, target, tokenizer.vocab_size)
            print(f"{rank}: {loss.item()}")
            loss.backward()

            acc_input_batches.append(inp_batch)

    # here we must wait that the data is properly received
    for req in reqs_wait:
        req.wait()

    # BACKWARD PASS:
    reqs_wait = []
    for bwd_iter in range(n_micro_batch):
        tag = bwd_iter + outer_iter
        if rank == 2 or rank == 5:
            # we need the activation from the previous batch
            out = acc_input_batches.pop()
            req = dist.isend(out.grad.to("cpu"), dst=rank - 1, tag=tag)
            reqs_wait.append(req)

        elif rank == 1 or rank == 4:
            # we wait for the activation
            inp_grad = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(inp_grad, src=rank + 1, tag=tag)
            req.wait()

            out = acc_outs.pop()
            out.backward(inp_grad.to(device))

            inp = acc_input_batches.pop()
            req = dist.isend(inp.grad.to("cpu"), dst=rank - 1, tag=tag)
            reqs_wait.append(req)

        elif rank == 0 or rank == 3:
            inp_grad = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(inp_grad, src=rank + 1, tag=tag)
            req.wait()

            out = acc_outs.pop()
            out.backward(inp_grad.to(device))

    # here we must wait that the data is properly received
    for req in reqs_wait:
        req.wait()

    # have all process wait here; we need to communicate and average the weights
    dist.barrier()

    tmp = []
    for param in net.parameters():
        if param.grad is None:
            tmp.append(torch.zeros_like(param).view(-1))
            continue

        tmp.append(param.grad.view(-1))
        param.grad = None

    prev_grad = torch.cat(tmp).to("cpu")
    if rank == 0 or rank == 3:
        dist.all_reduce(prev_grad, op=dist.ReduceOp.SUM, group=group_0)
    elif rank == 1 or rank == 4:
        dist.all_reduce(prev_grad, op=dist.ReduceOp.SUM, group=group_1)
    elif rank == 2 or rank == 5:
        dist.all_reduce(prev_grad, op=dist.ReduceOp.SUM, group=group_2)

    tmp = torch.split(prev_grad, len_sizes)
    for i, param in enumerate(net.parameters()):
        param.grad = tmp[i].view(sizes[i]).to(device) / 2  # average

    optim.step()
    torch.cuda.empty_cache()
