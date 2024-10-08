# GPT-2 (124M) repro on Lichess_UCI
# 124M parameter model on 10B tokens
# => 6 * 124e6 * 10e9 = 7.44e18 ~= 7e18 capability model
# 18,865 steps of 524,288 tokens/step
# on 1X RTX 3060m steps in ~300ms/iter
# => training time 18,865 * 300ms = 94.3 min ~= $20

make train_chesscu USE_CUDNN=1
out_dir="log_chess_gpt2_124M"
done_file="$out_dir/DONE_00018865"

# in case the training stalls or crashes, loop to resume (-y 1)
# while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi

    # run python dev/data/fineweb.py --version 10B to prepro data
    # run python dev/data/hellaswag.py to prepro hellaswag eval
    mpirun -np 1 ./train_chesscu \
                -i "dev/data/lichess_uci/00000000000000000010000/lichess_uci_train_*.bin" \
                -j "dev/data/lichess_uci/00000000000000000010000/lichess_uci_val_*.bin" \
                -o $out_dir \
                -v 250 \
                -s 20000 \
                -g 144 \
                -h 0 \
                -b 4 \
                -t 1024 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0006 \
                -q 0.0 \
                -u 700 \
                -n 5000 \
                -y 1 \
                -e "gpt2:d12"

                # -e "austindavis/chess-gpt2-uci-12x12x768"

    sleep 1
# done
