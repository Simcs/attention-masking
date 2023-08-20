for T in 0.1 0.2 0.4 0.8 1 2 4 8 16
do
    echo python -m bin.compute_ucf_hmdb_energy \
        --ucf_fold 1 --hmdb_fold 1 --T ${T}
    python -m bin.compute_ucf_hmdb_energy \
        --ucf_fold 1 \
        --hmdb_fold 1 \
        --T ${T}
done