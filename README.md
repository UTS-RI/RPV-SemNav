RPV-SemNav

Official repo for the paper "Room-Mediated Co-occurrence for Object-Centric Zero-Shot Semantic Navigation via Frontier Scoring", accepted to IROS 2026

TO DO:
- Add V1 code
- Clean-up unused files

## Installation

**This has been tested on Ubuntu 22.04 & 24.04**

1. **Install miniconda3**

   Follow the instructions at https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install

   Turn off automatic conda init to the `base` environment:
   ```bash
   conda config --set auto_activate_base false
   ```
   Note: you can undo this later by running `conda init --reverse $SHELL`.

2. **Create the conda environment with base packages**
   
   Inside the "env_installation_files" directory:
   ```bash
   conda env create -f environment-rpv.yml
   ```
   This creates an environment named `rpv`.

3. **Activate the created conda environment**
   ```bash
   conda activate rpv
   ```

4. **Install base Python packages**
   ```bash
   pip install -r requirements-rpv.txt
   ```

5. **Clone habitat-sim v0.3.3 into the project root directory**
   
   In the project root directory:
   ```bash
   git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-sim.git
   ```

6. **Install habitat-sim from source**

   Follow the steps at https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md

   You should not have to install habitat-sim's `requirements.txt`, as these requirements should already be covered by step 4. However, inside `habitat-sim`, you can check with:
   ```bash
   pip install -r requirements.txt --dry-run
   ```

   Also check for Linux system dependencies:
   ```bash
   sudo apt-get install -y --no-install-recommends \
     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
   ```

   If planning to install habitat-sim with CUDA compatibility, ensure the `CUDA_HOME` environment variable is set:
   ```bash
   export CUDA_HOME=$CONDA_PREFIX
   ```

   Inside `habitat-sim`, install with the desired configuration environment variables. We used a headless system with CUDA, and turned bullet physics on:
   ```bash
   HABITAT_BUILD_GUI_VIEWERS=OFF HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=ON \
     pip install . --no-build-isolation -c <path to requirements-new.txt from step 4>
   ```

7. **Verify the habitat-sim installation**

   To confirm the compiled bindings actually load without an `ImportError`, run:
   ```bash
   python -c "import habitat_sim; print(habitat_sim.__file__)"
   ```

   If compiling with `HABITAT_WITH_CUDA=ON`, verify the following prints `True`:
   ```bash
   python -c "import habitat_sim; print(habitat_sim.cuda_enabled)"
   ```
   If CUDA compatibility returns `False`, see [Common Installation Problems](#common-installation-problems) below.

8. **Install compatible habitat-lab and habitat-baselines versions with habitat-sim**
   ```bash
   pip install -r requirements-habitat.txt -c requirements-new.txt
   ```

9. **Fix syntax errors in habitat-lab/habitat-baselines with Python 3.12**
   ```bash
   python fix_habitat_python312_new.py
   ```
   Confirm the fix with:
   ```bash
   python -c "import habitat; import habitat_baselines; print('habitat-lab OK, version:', habitat.__version__ if hasattr(habitat, '__version__') else 'imported'); print('habitat-baselines OK')"
   ```

10. **Install editable versions of vlfm and frontier_exploration** (updated to work with Habitat v0.3.X)

    From the project root folder:
    ```bash
    pip install -e ./frontier_exploration -e ./vlfm -c <path to requirements-new.txt>
    ```

11. **Install detectron2**
    ```bash
    pip install --no-build-isolation --no-deps \
      "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@fd27788985af0f4ca800bca563acdb700bb890e2"
    ```
    This must be its own step, as `--no-build-isolation` is needed. Detectron2's `setup.py` imports torch directly to detect your CUDA compute capability and compile custom ops, so it needs to see the already-installed torch, not a fresh isolated build env.

    To confirm the installation, check:
    ```bash
    python -c "import detectron2; import iopath; print('detectron2 OK, iopath version:', iopath.__version__)"
    ```
    Expected output: `detectron2 OK, iopath version: 0.1.10`

12. **Install Mask2Former and the CUDA kernel for MSDeformAttn**

    From the repo root directory, following https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md (it should not be necessary to install Mask2Former's `requirements.txt`):
    ```bash
    git clone https://github.com/facebookresearch/Mask2Former.git
    cd Mask2Former
    pip install -r requirements.txt -c <path to requirements-new.txt>
    cd mask2former/modeling/pixel_decoder/ops
    ```

    Mask2Former's custom CUDA kernel (`MultiScaleDeformableAttention`) was written against an older PyTorch API and fails to compile against modern PyTorch (2.x) with an error like:
    ```
    error: no suitable conversion function from "const at::DeprecatedTypeProperties" to "c10::ScalarType" exists
    ```
    This happens because the kernel calls `AT_DISPATCH_FLOATING_TYPES(value.type(), ...)`, and PyTorch removed the implicit `Tensor.type() -> ScalarType` conversion this relies on. Before building, patch the kernel source to use the modern equivalents:
    ```bash
    sed -i 's/value\.type()\.is_cuda()/value.is_cuda()/g' src/cuda/ms_deform_attn_cuda.cu
    sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type(),/AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),/g' src/cuda/ms_deform_attn_cuda.cu
    sed -i 's/value\.type()\.is_cuda()/value.is_cuda()/g' src/ms_deform_attn.h
    ```

    Then build as normal:
    ```bash
    CUDA_HOME=$CONDA_PREFIX sh make.sh
    ```

    Verify it built correctly, running the following commands from the `Mask2Former` directory:
    ```bash
    python -c "from mask2former.modeling.pixel_decoder.ops.functions import MSDeformAttnFunction; print('MSDeformAttn OK')"
    python -c "import torch; import MultiScaleDeformableAttention; print('Compiled kernel loaded OK')"
    ```

    **Overall installation check** (from the repo root directory):
    ```bash
    python -c "
    import torch
    import habitat_sim
    import habitat
    import habitat_baselines
    import detectron2
    import sam3
    import vlfm
    import frontier_exploration
    from Mask2Former.mask2former.modeling.pixel_decoder.ops.functions import MSDeformAttnFunction
    print('torch CUDA available:', torch.cuda.is_available())
    print('habitat_sim CUDA enabled:', habitat_sim.cuda_enabled)
    print('All imports OK')
    "
    ```

## Common Installation Problems

### Step 7: `habitat_sim.cuda_enabled` returns `False`

If CUDA compatibility returns `False` after building habitat-sim, force a clean rebuild:

1. Uninstall habitat-sim, clear any leftover CMake/scikit-build build directories from the prior installation, purge the pip cache, and make sure the conda environment is activated with `CUDA_HOME` set:
   ```bash
   pip uninstall habitat_sim -y
   rm -rf build _skbuild *.egg-info
   pip cache purge
   export CUDA_HOME=$CONDA_PREFIX
   ```

2. Reinstall with explicit CMake args:
   ```bash
   CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_WITH_CUDA=ON" \
   HABITAT_SIM_HEADLESS=1 \
   HABITAT_BUILD_GUI_VIEWERS=OFF \
   HABITAT_WITH_CUDA=ON \
   HABITAT_WITH_BULLET=ON \
     pip install . --no-build-isolation -c <path to requirements-new.txt from step 4> -v
   ```

   OR, using the legacy (v0.3.3) env var names directly:
   ```bash
   HEADLESS=True \
   WITH_CUDA=True \
   WITH_BULLET=True \
     pip install . --no-build-isolation -c <path to requirements-new.txt from step 4> -v
   ```

   **Note:** The official `BUILD_FROM_SOURCE.md` on habitat-sim's `main` branch documents the `HABITAT_WITH_CUDA` / `HABITAT_BUILD_GUI_VIEWERS` env vars for a newer scikit-build-core-based build system. `v0.3.3` predates that migration and uses a legacy `setup.py` that reads different, unprefixed variable names: `WITH_CUDA`, `HEADLESS`, `WITH_BULLET`. Using `main`'s documented variable names against this tag will silently build **without** CUDA — pip reports success either way, since the wrong env var name is just ignored, not rejected. This is the most common cause of step 7 failing.

   Full sequence for reference:
   ```bash
   export CUDA_HOME=$CONDA_PREFIX
   git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-sim.git
   cd habitat-sim
   HEADLESS=True WITH_CUDA=True WITH_BULLET=True \
     pip install . --no-build-isolation -c <path to requirements-new.txt from step 4> -v
   ```
