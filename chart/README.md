# sys-stats Helm chart

Deploys the [sys-stats](https://github.com/obeone/sys-stats) server (Flask API +
web UI) on Kubernetes, built on the
[bjw-s common library](https://github.com/bjw-s-labs/helm-charts) — everything is
configured through `values.yaml`, the chart itself carries no templates of its
own.

## Install

```bash
helm dependency update chart/
helm upgrade --install sys-stats ./chart --namespace monitoring --create-namespace
```

## What it needs from the node

The dashboard reports on the machine the pod lands on, so the defaults are
deliberately less locked down than a normal web app would be:

| Setting                            | Default | Why                                                                  |
| ---------------------------------- | ------- | -------------------------------------------------------------------- |
| `defaultPodOptions.hostPID`         | `true`  | Without the host PID namespace the process tables show one process.   |
| `containers.main.securityContext.privileged` | `true` | Reading other users' `/proc` entries and reaching `nvidia-smi`. |
| `replicas`                          | `1`     | Several replicas would each report on a different node at random.     |

A pod that floats between nodes gives a dashboard that silently changes subject.
Pin it:

```yaml
defaultPodOptions:
  nodeSelector:
    kubernetes.io/hostname: gpu-node-01
```

## Common tweaks

### GPU panels

They need a card claimed through the
[NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin). It is
commented out in `values.yaml` because the request makes the pod unschedulable
on a node without one:

```yaml
controllers:
  main:
    containers:
      main:
        resources:
          limits:
            nvidia.com/gpu: 1
```

Clusters that expose the devices through a RuntimeClass also need
`defaultPodOptions.runtimeClassName: nvidia`.

### Ollama panel

Unset, the server answers with an empty model list rather than an error, and the
panel stays empty:

```yaml
controllers:
  main:
    containers:
      main:
        env:
          OLLAMA_API_URL: http://ollama.ollama.svc.cluster.local:11434
```

### Exposing it

```yaml
ingress:
  main:
    enabled: true
    className: nginx
    hosts:
      - host: sys-stats.example.com
        paths:
          - path: /
            pathType: Prefix
            service:
              identifier: main
              port: http
    tls:
      - secretName: sys-stats-tls
        hosts:
          - sys-stats.example.com
```

Gateway API users can drop the ingress and use the library's `route` block
instead.

## Version

`appVersion` in `Chart.yaml` is the only place the deployed sys-stats version is
written down — `image.tag` renders it through `tpl`. Bump one, not two.

Anything the common library accepts works here; its
[values reference](https://github.com/bjw-s-labs/helm-charts/blob/main/charts/library/common/values.yaml)
is the full list.
