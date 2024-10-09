# GPT-2 (124M) repro on FineWeb
# 124M parameter model on 10B tokens
# => 6 * 124e6 * 10e9 = 7.44e18 ~= 7e18 capability model
# 18,865 steps of 524,288 tokens/step
# on 8X A100 80GB SXM ($14/hr) steps in ~300ms/iter
# => training time 18,865 * 300ms = 94.3 min ~= $20

make train_gpt2cu USE_CUDNN=1
out_dir="log_gpt2_124M"
done_file="$out_dir/DONE_00018865"

# in case the training stalls or crashes, loop to resume (-y 1)
while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi
    # Notes on data prep:
    # - run python dev/data/fineweb.py -t edu to prepare data
    # - I limited the download to the first few by first downloading
    #   the README and several of the parquet files. Then I edited the
    #   README to add a new configuration called 'subsample' that 
    #   included only the parquet files that I had already downloaded. 

    # run python dev/data/hellaswag.py to prepro hellaswag eval

    # Notes on parameters: 
    # d % (b*t) must equal zero
    # RTX 3060 mobile can handle b=8, t=1024, d=524288
    # Set n=100 to checkpoint model in case of crash
    mpirun -np 1 ./train_gpt2cu \
                -i "dev/data/fineweb10B/fineweb_train_*.bin" \
                -j "dev/data/fineweb10B/fineweb_val_*.bin" \
                -o $out_dir \
                -v 250 -s 20000 -g 144 \
                -h 1 \
                -b 8 -t 1024 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0006 \
                -q 0.0 \
                -u 700 \
                -n 100 \
                -y 1 \
                -e "d12"

    sleep 1
done
