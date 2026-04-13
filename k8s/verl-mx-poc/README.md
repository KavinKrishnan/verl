# verl + ModelExpress POC (GKE GB200)

## Prerequisites

- MX Server + Redis running in `kavin` namespace
- `shared-model-cache` PVC with HF model cache
- `nvcr-imagepullsecret` for NGC registry
- `kavin-compute-domain` ComputeDomain with channel template

## Deploy

```bash
# 1. Config
kubectl apply -f config.yaml

# 2. Ray head (runs trainer + launches verl PPO)
kubectl apply -f ray-head.yaml

# 3. Watch logs
kubectl logs -f verl-mx-head-0 -n kavin
```

## Config

Set `checkpoint_engine.backend = "mx"` in the Hydra overrides (see `config.yaml`).

The MX checkpoint engine uses:
- NIXL RDMA for GPU-to-GPU weight transfer
- MX Server for metadata coordination
- Star topology: trainer rank 0 → all rollout ranks

## Notes

- Single-node mode (4 GPUs): trainer and rollout are colocated Ray actors.
  For disaggregated mode, add a separate ray-worker StatefulSet on a second node.
- The `compute-domain-channel` resource claim is required for cross-node GPU RDMA.
