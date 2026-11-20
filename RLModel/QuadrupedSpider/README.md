# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/QuadrupedSpider

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: If the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Train the privileged teacher and belief-state student

Train a new teacher first. The teacher task observes proprioception, foot-contact points, and
a persistent terrain map that combines the ground-only height scan with contact corrections.
It writes runs under `logs/rsl_rl/QuadrupedSpider-PrivilegedTeacher`:

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-QuadrupedSpider-Teacher-v0 \
    --run_name privileged
```

After the teacher has converged, train a fresh student. The training script selects the
latest teacher checkpoint whose input matches the current proprioception-plus-privileged
observation size; a proprioception-only legacy checkpoint is rejected:

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-QuadrupedSpider-Student-v0 \
    --run_name belief_student
```

The sensor casts a 31x31, 2.4m x 2.4m grid against `/World/ground` only. Contact sensors do not
make objects visible to that scanner: they report foot contact locations and forces. The
environment registers the scan in a rolling world-coordinate map, inserts measured contact
heights, and retains those contact-derived cells as the robot moves. The map is cleared on
episode reset. During teacher and student training, half of episode resets place one
collision-enabled 5, 10, or 15 cm block in the robot's path; these blocks are not included in the
raycaster targets. Play scripts disable these training props so a separately placed,
sensor-invisible cube remains a held-out test.

The student receives this full grid with simulated noise, dropouts, and blackouts, plus
proprioception including joint effort (so contact cues can be estimated from onboard-like signals,
not simulator contact labels). A recurrent encoder maintains the student's
belief and decodes a full-grid perceived map. Distillation trains its actions against the
teacher and its map against the persistent contact-corrected target. In `controll.py`, red dots
show the current raw ground scan and green dots show the student's predicted map. An obstacle
that the scanner cannot see should therefore appear in green only after the foot contacts it,
and should remain in the map after the foot leaves.

Both teacher and student checkpoints must be retrained after this observation/model change.
Older checkpoints are not compatible with the new input and output dimensions. The contact-map
behavior is an adaptation inspired by Miki et al.; it still requires a live simulation benchmark
with an opaque-to-scan obstacle to verify contact height accuracy and retention.

### Setup as Omniverse Extension (Optional)

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/QuadrupedSpider"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```