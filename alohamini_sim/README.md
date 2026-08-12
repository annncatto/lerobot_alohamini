# alohamini_sim

Simulation stack for AlohaMini: turn a phone video of a real room into a photoreal
Isaac Sim scene, generate scripted manipulation episodes in it, and export them as a
[LeRobotDataset](../src/lerobot/datasets/lerobot_dataset.py) that co-trains with real
AlohaMini recordings.

```
alohamini_sim/
├── video2sim/     # phone video → NuRec splat room + TSDF collider in Isaac Sim 5.x
└── data_engine/   # scripted sim episode engine + LeRobotDataset bridge
    ├── data_gen/                    # vendored AlohaMini/maniskill/data_gen (original layout)
    │   ├── aspire_engine/           #   engine.py, skills_runtime.py, planner + repair loop
    │   ├── intern_engine/           #   InternDataEngine-style pipeline (skills, planners, writers)
    │   └── tasks.py                 #   ManiSkill env registration (AlohaMiniMultiYCB-v1, ...)
    ├── agents/aloha_mini/           # vendored ManiSkill agent (SO100 + Pro parallel gripper)
    │   └── assets/                  #   robot URDFs + STL meshes the agents load
    └── lerobot_bridge.py            # episodes → this repo's LeRobotDataset (v3.0)
```

## Two environments, on purpose

- **`video2sim/` and the engine run in an external GPU toolchain** (Isaac Sim 5.x,
  LingBot-Map, gsplat, ...), **not** in this repo's uv environment. That stack documents
  and validates itself: see [`video2sim/README.md`](video2sim/README.md) and run
  `python -m video2sim.check_env` inside that environment.
- **`data_engine/lerobot_bridge.py` and its tests run in this repo's uv environment.**
  The bridge only needs numpy + the in-repo dataset API, so converted episodes are
  written and validated with the exact same code path real recordings use.

## data_engine layout

`data_engine/` is three layers with different dependency budgets:

1. **Planner/skills (vendored, executor-only).** `data_gen/` is a verbatim vendored copy
   of `AlohaMini/maniskill/data_gen` (aspire_engine + intern_engine + tasks.py) and
   `agents/aloha_mini/` contains the maintained `AlohaMini2ProSim` Agent plus the legacy
   `AlohaMiniSO100V2`/`AlohaMiniProV2`/`AlohaMiniProV3` adapters. The original
   top-level import style (`import data_gen.aspire_engine.engine`, `import agents.aloha_mini`)
   still works: `aspire_engine/engine.py` inserts `alohamini_sim/data_engine` into
   `sys.path` (add it yourself before the first import, as
   `tests/test_alohamini_sim_engine_imports.py` does). **Executing this layer needs the
   external ManiSkill toolchain, which is not part of this repo's uv env:**

   ```bash
   pip install mani-skill sapien torch   # plus a GPU + Vulkan for actual rollouts
   ```

2. **Bridge (runs in this repo's uv env).** `lerobot_bridge.py` +
   `data_gen/aspire_engine/writer_adapter.py` only need numpy and the in-repo dataset
   API. They must stay importable without `mani_skill`; `data_gen/__init__.py` is kept
   dependency-free on purpose (env registration happens inside the executor modules),
   and `tests/test_alohamini_sim_engine_imports.py` guards this.

3. **Assets (runtime mirror).** The canonical description, documentation, calibration
   notes, and asset tests live in RoboTwin at
   `assets/embodiments/alohamini2pro/`. This directory mirrors only runtime files
   (URDF/SRDF, meshes, YAML, and the SAPIEN adapter):

   ```bash
   python alohamini_sim/scripts/sync_alohamini2pro_asset.py --sync
   python alohamini_sim/scripts/sync_alohamini2pro_asset.py
   ```

   The first command copies missing/changed runtime files without deletion; the second
   is a read-only drift check. README/TODO/reference/test files are deliberately not
   mirrored. `AlohaMini2ProSim` loads this 29-link, 18-DoF asset directly.
   Run the headless loading/control gate in the ManiSkill environment with:

   ```bash
   conda run -n maniskill python alohamini_sim/scripts/smoke_alohamini2pro.py
   ```

   The Isaac-side robot for video2sim is separate:
   [`video2sim/assets/am2pro_parallel/alohamini2pro_parallel.urdf`](video2sim/assets/am2pro_parallel/alohamini2pro_parallel.urdf)
   it must not be used as the canonical dynamics/collision description.

## The bridge

`lerobot_bridge.py` converts current engine episodes (per-step 18-D `qpos`, 18-D controller
targets, optional per-camera uint8 RGB) into a v3.0 LeRobotDataset whose feature names
follow the AlohaMini2Pro convention (including wrist yaw, `x.vel`,
`y.vel`, `theta.vel`, `lift_axis.height_mm`, `observation.images.<cam>`), so sim data
can be mixed with `lerobot-record` output. See the module docstring for the exact
per-dimension value mapping (base positions → body-frame velocities, lift → mm,
optional gripper 0-100 scaling).

The bridge selects the layout from each episode's action width: current
`alohamini2pro_sim` episodes use 18-D actions and the HD 6-DoF feature profile;
previously generated 16-D/5-DoF episode pickles remain readable for migration.

### Current validation boundary

- Asset drift check, 29-link/18-DoF loading, 16 kg runtime mass, fixed-base control,
  current/legacy dataset conversion, and the Pinocchio IK station gate are validated.
- The old `AlohaMiniMultiYCB-v1` scripted grasp trajectory was written for the legacy
  gripper geometry. It now runs without the previous live-qpos finite-difference IK
  failure, but seed 0 still pushes the Rubik's cube instead of lifting it. Do not use
  that task to generate training data until its contact trajectory is retuned and a
  multi-seed grasp gate passes. The maintained Agent/asset smoke test above is the
  current acceptance gate.

As a library:

```python
from alohamini_sim.data_engine.lerobot_bridge import write_episodes

write_episodes(episodes, repo_id="local/alohamini_sim_pick", root="out/lerobot_ds", fps=20)
```

As a CLI (episodes pickled as a list of engine episode dicts):

```bash
python -m alohamini_sim.data_engine.lerobot_bridge \
    --episodes out/episodes.pkl --repo-id local/alohamini_sim_pick --root out/lerobot_ds
```

## Quickstart: run the bridge tests

```bash
uv sync --locked --extra test --extra dataset   # dataset extra is required by the bridge
uv run pytest tests/test_alohamini_sim_bridge.py tests/test_alohamini_sim_engine_imports.py -svv
```

`test_alohamini_sim_engine_imports.py` additionally asserts the bridge imports without
`mani_skill`; its executor-side test (importing `data_gen.aspire_engine.engine`
end-to-end) is skipped unless `mani_skill` is installed.

The tests build tiny synthetic episodes (10 frames, two 96x96 cameras), write a dataset
into a temp directory, reload it with `LeRobotDataset`, and assert feature names, dtypes,
and joint/pixel/task round-trips — including the `python -m` CLI path.

For the full room pipeline (capture checklist, stage-by-stage commands, GPU/memory
requirements), start at [`video2sim/README.md`](video2sim/README.md).
