f="$1"
r="$2"
devices="$3"

python3 main.py \
        --base configs/calvin_vqgan_f"$f"_r"$r".yaml \
        -t True \
        -n calvin-vqgan-f"$f"-r"$r" \
        --accelerator gpu \
        --devices "$devices" \
        --max_epochs 50 \
        --no-test True