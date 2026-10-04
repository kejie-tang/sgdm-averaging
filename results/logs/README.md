# Experiment logs

Console logs from the authors' runs are stored here, e.g.

```
bash scripts/run_all.sh > results/logs/full_run.log 2>&1
```

This is useful when re-running the full experiments is impractical: a reviewer
can compare their own output against the stored raw arrays (`*.npz` in each
`results/<exp>/` directory) and these logs to verify the reported results.
