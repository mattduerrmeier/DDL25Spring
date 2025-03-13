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

rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
world_size = 3
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6 // world_size
seq_l = 256
batch_size = 3
device = "cuda"


# make the model
if rank == 0:
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
elif rank == 1:
    net = LLamaStage(
        dmodel=dmodel,
        num_heads=num_heads,
        device=device,
        n_layers=n_layers,
        ctx_size=seq_l,
    )
elif rank == 2:
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

n_micro_batch = 3
micro_batch_size = batch_size // n_micro_batch

for outer_iter in range(5_000):
    # FORWARD PASS:
    optim.zero_grad()
    reqs_wait = []
    if rank == 0:
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

            req = dist.isend(out.to("cpu"), dst=1, tag=tag)
            reqs_wait.append(req)

    elif rank == 1:
        # accumulate the inputs and activations
        acc_outs = []
        acc_input_batches = []

        for iter_micro_batch in range(n_micro_batch):
            tag = iter_micro_batch + outer_iter
            inp_batch = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(inp_batch, src=0, tag=tag)
            req.wait()

            with torch.no_grad():
                inp_batch = inp_batch.to(device)
                inp_batch.requires_grad_()
                inp_batch.retain_grad()

            out = net(inp_batch)
            acc_outs.append(out)
            acc_input_batches.append(inp_batch)

            req = dist.isend(out.to("cpu"), 2, tag=tag)
            reqs_wait.append(req)

    elif rank == 2:
        # accumulate the inputs
        acc_input_batches = []

        target_mini_batch = next(iter_ds)
        target_micro_batches = torch.chunk(target_mini_batch, n_micro_batch)

        for iter_micro_batch, target in enumerate(target_micro_batches):
            tag = iter_micro_batch + outer_iter
            inp_batch = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(inp_batch, src=1, tag=tag)
            req.wait()

            with torch.no_grad():
                inp_batch = inp_batch.to(device)
                inp_batch.requires_grad_()
                inp_batch.retain_grad()

            logits = net(inp_batch)
            loss = causalLLMLoss(logits, target, tokenizer.vocab_size)
            print(loss.item())
            loss.backward()

            acc_input_batches.append(inp_batch)

    # here we must wait that the data is properly received
    for req in reqs_wait:
        req.wait()

    # BACKWARD PASS:
    reqs_wait = []
    for bwd_iter in range(n_micro_batch):
        tag = bwd_iter + outer_iter
        if rank == 2:
            # we need the activation from the previous batch
            out = acc_input_batches.pop()
            req = dist.isend(out.grad.to("cpu"), dst=1, tag=tag)
            reqs_wait.append(req)

        elif rank == 1:
            # we wait for the activation
            inp_grad = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(inp_grad, src=2, tag=tag)
            req.wait()

            out = acc_outs.pop()
            out.backward(inp_grad.to(device))

            inp = acc_input_batches.pop()
            req = dist.isend(inp.grad.to("cpu"), 0, tag=tag)
            reqs_wait.append(req)

        elif rank == 0:
            inp_grad = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(inp_grad, src=1, tag=tag)
            req.wait()

            out = acc_outs.pop()
            out.backward(inp_grad.to(device))

    # here we must wait that the data is properly received
    for req in reqs_wait:
        req.wait()

    optim.step()
    torch.cuda.empty_cache()

    # abele.malan@unine.ch
