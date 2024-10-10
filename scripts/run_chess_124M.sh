# GPT-2 (124M) repro on Lichess 201506-moves
# 124M parameter model on 10B tokens
# => 6 * 124e6 * 10e9 = 7.44e18 ~= 7e18 capability model
# 18,865 steps of 524,288 tokens/step
# on 8X A100 80GB SXM ($14/hr) steps in ~300ms/iter
# => training time 18,865 * 300ms = 94.3 min ~= $20

make train_chesscu USE_CUDNN=1
out_dir="log_chess_gpt"
done_file="$out_dir/DONE_00018865"

# in case the training stalls or crashes, loop to resume (-y 1)
while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi
    # Notes on data prep:
    # - run python dev/data/lichess_uci.py -d 201506 to prepare data
    # - I limited the download to the first few by first downloading
    #   the README and several of the parquet files. Then I edited the
    #   README to add a new configuration called 'subsample' that 
    #   included only the parquet files that I had already downloaded. 

    # Notes on parameters: 
    # d % (b*t) must equal zero
    # RTX 3060 mobile can handle:
    #    d8: b=14
    #    d12: b=38
    # Set n=100 to checkpoint model in case of crash
    mpirun -np 1 ./train_chesscu \
                -e "chessGPT_d8_bf16.bin" 
                -i "dev/data/201506-moves/201506_train_*.bin" 
                -j "dev/data/201506-moves/201506_val_*.bin" 
                -lg 1 -n 0 -o $out_dir -y 1 \
                -b 35 -t 1024 \
                -g 64 -s 20  -v 250\
                -c 0.1 -l 0.0006 -q 0.0 -u 700 \
                -r 0 -z 1 

    sleep 1
done
