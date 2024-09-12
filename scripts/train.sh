f="$1"
devices="$2"
python3 main.py \
        --base configs/calvin_vqgan_f"$f".yaml \
        -t True \
        -n calvin-vqgan-f"$f" \
        --accelerator gpu \
        --devices "$devices" \
        --max_epochs 20