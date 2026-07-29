# Repository instructions

## Safety rules

- Only modify files inside this repository.
- Do not access, create, modify, move, or delete files outside this repository.
- Reading or executing the explicitly approved project Python interpreter outside
  the repository is allowed, but do not modify the environment unless explicitly
  requested.
- Do not use sudo.
- Do not modify `/usr`, `/etc`, `/opt`, system services, NVIDIA drivers, CUDA,
  ROS installations, system Python, or shell configuration files.
- Do not modify the Conda base environment.
- Do not install packages globally.
- Do not install, upgrade, downgrade, or uninstall packages unless explicitly
  requested by the user.
- When package installation is explicitly requested, install packages only into
  the approved project environment.
- Before deleting files, overwriting datasets, removing environments, or making
  destructive changes, ask for approval.
- Do not delete datasets, checkpoints, experiment results, logs, or annotations.
- Record project dependencies in `requirements.txt`, `pyproject.toml`, or
  `environment.yml`.
- Do not change documented annotation conventions or tensor conventions without
  explicit approval.

## Python environment

Use this exact Python interpreter for this repository:

```text
/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/conda_env/bin/python
```

For every Python command, invoke that interpreter by its absolute path.

Do not use:

- `python`
- `python3`
- `/usr/bin/python3`
- the Conda base environment
- another automatically discovered Python interpreter

Before running tests or Python scripts, verify the environment with:

```bash
PROJECT_PYTHON=/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/conda_env/bin/python

"$PROJECT_PYTHON" - <<'PY'
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

If the interpreter does not exist, or importing `torch` fails:

- stop;
- report the problem;
- do not fall back to `python3`, system Python, or another environment;
- do not skip PyTorch-dependent tests.

## Testing rules

Run pytest with external plugin autoload disabled:

```bash
PROJECT_PYTHON=/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/conda_env/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
"$PROJECT_PYTHON" -m pytest -q
```

Run syntax checks with:

```bash
"$PROJECT_PYTHON" -m compileall src scripts tests
```

Also run:

```bash
git diff --check
```

A test round is complete only when:

- all requested tests actually run;
- there are zero failures;
- there are zero unexpected skips;
- `compileall` passes;
- `git diff --check` passes.

Do not describe skipped PyTorch tests as successful validation.

## Development rules

- Work on only the scope requested in the current prompt.
- Do not begin the next implementation round automatically.
- Do not silently weaken validation rules to make tests pass.
- Determine whether a failure comes from the implementation or the test before
  changing anything.
- Preserve these project invariants:
  - heatmap samples use shape `[3, H, T]`;
  - batched heatmaps use shape `[B, 3, H, T_max]`;
  - temporal width is never resized;
  - padding is applied only on the right;
  - padded labels use `-100`;
  - padded positions are excluded using a boolean valid mask;
  - timestamp segment ends are exclusive;
  - frame segment ends are inclusive;
  - output timestep `t` corresponds to heatmap column `t`.
- Do not commit changes unless explicitly requested.
