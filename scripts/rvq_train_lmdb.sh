ch_mult="$1" # 11224
devices="$2" # 0,1,2,3
codebook_size="$3" # 1024
codebook_dim="$4" # 256
num_quantizers="$5" # 4

python3 main.py \
        --base configs/calvin_rvqgan_"$ch_mult"_lmdb.yaml \
        -t True \
        -n calvin-rvqgan \
        --accelerator gpu \
        --devices "$devices" \
        --max_epochs 30 \
        --no-test True \
        --codebook_size "$codebook_size" \
        --codebook_dim "$codebook_dim" \
        --num_quantizers "$num_quantizers" \
        --postfix "_ch${ch_mult}_cs${codebook_size}_cd${codebook_dim}_nq${num_quantizers}" \