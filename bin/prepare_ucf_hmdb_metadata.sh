for split in train test
do
    for fold in 1 2 3
    do
        echo python -m bin.prepare_ucf_metadata \
            --split ${split} --fold ${fold}
        python -m bin.prepare_ucf_metadata \
            --split ${split} \
            --fold ${fold}
    done
done

for split in train test
do
    for fold in 1 2 3
    do
        echo python -m bin.prepare_hmdb_metadata \
            --split ${split} --fold ${fold}
        python -m bin.prepare_hmdb_metadata \
            --split ${split} \
            --fold ${fold}
    done
done