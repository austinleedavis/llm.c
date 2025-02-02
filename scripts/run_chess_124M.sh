# ChessGPT pretrain on Lichess 201506-moves
# Training time limited by -x:
# 5957028 instances 
# @b=11: 541548 steps: 216000 max steps at 24hrs 

# kill any lingering chess_cu training processes:
nvidia-smi | grep .train_ | awk '{print $5}' | xargs -I {} kill -9 {}

make train_chesscu USE_CUDNN=1
depth="d12"
out_dir="log_chess_gpt_$depth"
done_file="$out_dir/DONE_00541548"
train_date="202401"
val_date="201301"

# run the python file to ensure the model.bin is available
# python train_chess.py --model $depth

# in case the training stalls or crashes, loop to resume (-y 1)
while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi
    # Notes on data prep:
    # - run python dev/data/lichess_uci.py -d 201701 to prepare data

    # Notes on parameters: 
    # d % (b*t) must equal zero
    # RTX 3060 mobile can handle:
    #    d8 use:  b=28 @ 380ms/step -> 8 hours -> -x 75800 max steps
    #    d12 use: b=11 @ 400ms/step -> 8 hrs  -> -x 72000 max steps
    # Set n=100 to checkpoint model in case of crash
    mpirun -np 1 ./train_chesscu \
                -e "model_00216000.bin" \
                -i "dev/data/${train_date}-moves/*_train_*.bin" \
                -j "dev/data/${val_date}-moves/*_val_*.bin" \
                -lg 500 -n 500 -nk 1 -o $out_dir -y 1 \
                -b 11 -t 1024 \
                -g 640 -s 500  -v 500\
                -c 0.1 -l 0.00007 -q 0.0 \
                -r 0 -z 1 
    sleep 1
done
