# scripts

These shell scripts hold the exact commands to llm.c that reproduce the GPT-2 and GPT-3 runs.

### pytorch reference runs

For all pyrun scripts, current restrictions:

- does not write checkpoint, only logs of the train/val losses
- does not evaluate hellaswag accuracy
- cannot "resume training" (i.e. the `-y 1` flag)

### memory considerations

In any of these scripts, if you are running out of memory on your GPU you'll want to meddle with two flags: the recompute setting `-r` and the microbatch size `-b`. Recompute throws away some activations during the forward pass and then recomputes them during the backward pass. This reduces the amount of memory we need to store and cache during the forward pass, but then increases the amount of computation we need to do during the backward pass. The microbatch size controls the number of token streams that are processed in a single forward/backward pass in parallel. Decreasing this number means we need to store less memory per microbatch, but then we have to increase the number of loops in the gradient accumulation to meet the same desired total batch size.

Long story short, try `-r 1` (recompute GeLU, trading off speed and memory) to conserve some memory. If that doesn't help, start dividing the micro batch size until things fit. For example if the deafult is `-b 64`, try `-b 32`, and then 16, 8, etc. until things fit. Once they do fit, experiment with dialing back the recompute flag `-r 0` to get some speed back. Alternatively to `-b`, if your application doesn't need a very long context length, you can dial back the number of max sequence length using `-t`. For example GPT-2 uses `-t 1024` and GPT-3 uses `-t 2048`. Your application may tolerate a lower context length.

### multi-gpu considerations

It might be that you only have one GPU and not a whole box of them. Every script is fairly easy to change for just a single GPU. For llm.c, simply change line 1 to line 2 and leave everything else the same:

```bash
mpirun -np 8 ./train_gpt2cu \
./train_gpt2cu \
```

For PyTorch, the same thing:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py \
python train_gpt2.py \
```

Both of these scripts automatically detect how many GPUs are available and adjust the gradient accumulation inner loop of the optimization accordingly, so the results come out the same, up to floating point error. Of course, you'll have to wait proportionally longer for the optimization to finish.

To run on multiple nodes of GPUs, have a look at this pending [PR](https://github.com/karpathy/llm.c/pull/426), alternatively for llm.c try something like this:

```bash
mpirun -np 16 --host node1:8,node2:8 ./train_gptcu ...
```

For PyTorch follow the torchrun docs.

### Arguments:
<pre>
Usage:   ./train_gpt2cu [options]
Options:
  --help      Display this message.
 > File system input /
  -e <string> input .bin filename or descriptor, see code comments as docs. (default = chessGPT_d8_bf16.bin)
  -i <string> train data filename pattern (default = dev/data/201506/201506_train_*.bin)
  -j <string> val data filename pattern (default = dev/data/201506/201506_val_*.bin)
  -lg <int>   log gpu info every x steps (default = -1; disabled)
  -n <int>    write optimization checkpoints every how many steps? (default 0, don't)
  -nk <int>   max number of checkpoints to keep in the directory, removing old ones (0 = disable, default)
  -nm <int>   every how many step checkpoints are considered major? major checkpoints never get deleted.
  -o <string> output log dir (default = log_chess_gpt, no logging)
  -y <int>    resume optimization found inside output log dir? (0=restart/overwrite, 1=resume/append)
 > Token layout for each step of optimization
  -b <int>    (per-GPU, micro) batch size B (default = 4)
  -d <int>    total desired batch size (default = B * T * num_processes, i.e. no grad accumulation
  -t <int>    sequence length T (default = 1024)
 > Workload (number of steps)
  -x <int>    max_steps of optimization to run (-1 (default) = disable, run 1 epoch)
 > Optimization
  -c <float>  weight decay (default = 0.0f)
  -k <string> learning rate scheduler (default = cosine)
  -l <float>  learning rate (default = 3e-4f)
  -q <float>  learning rate decay: final fraction, at end of training (default = 1.0 (no decay))
  -sg <float> outlier stability: skip update if grad_norm goes above this in zscore (0.0f=off)
  -sl <float> outlier stability: skip update if loss goes above this in zscore (0.0f=off)
  -u <int>    learning rate warmup iterations (default = 0, no warmup)
 > Evaluation
  -g <int>    genT, how many steps of inference we do (default = 64)
  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)
  -s <int>    sample_every, how often we inference the model (default = 20)
  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)
 > Debugging
  -a <int>    overfit a single batch? 0/1. useful for debugging
 > Numerics
  -f <int>    enable_tf32 override (default: 1, set to 0 to disable tf32)
  -ge <int>   gelu fusion: 0=none, 1=forward, 2=forward+backward (default: 2 for >=SM90, 0 for older GPUs)
  -w <int>    keep f32 copy of weights for the optimizer? (default: 1)
 > Memory management
  -r <int>    recompute: less memory but less speed. (default = 1), 0|1|2 = none,gelu,gelu+ln
  -z <int>    zero_stage, Zero Optimization Stage, 0,1,2,3 (default = 0)
 > Multi-node settings
  -pg <int>    gpus_per_node (default = 8)
  -pm <string> nccl_init_method: tcp,fs,mpi (default = mpi)
  -pn <int>    num_processes (default = 1)
  -pp <string> fs_path - used only when nccl_init_method is fs (default = /tmp)
  -pr <int>    process_rank (default = 0)
  -ps <string> server_ip - used only when nccl_init_method is tcp (default = -1)
</pre>