for T in 1.5
do
    echo python -m bin.kinetics.compute_kinetics_energy \
        --T ${T}
    python -m bin.kinetics.compute_kinetics_energy \
        --T ${T}
done