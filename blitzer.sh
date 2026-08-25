for s in 42 43 44 45 46; do
  python -m reproduce_experiments.blitzer --seed $s
done