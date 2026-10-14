# Results and experiment logs

Each experiment writes into its own `results/<exp>/` directory:

* the **figure** (`*.png`) that appears in the paper — tracked in this
  repository so that the published panels can be inspected without re-running
  anything;
* the **raw arrays** (`*.npz`) behind that figure — not tracked (they are
  regenerable and sometimes large).

Console logs from the authors' runs are stored here, e.g.

```
bash scripts/run_all.sh > results/logs/full_run.log 2>&1
```

This is useful when re-running the full experiments is impractical: a reviewer
can compare their own output against the stored figure and the log of the
authors' run.
