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

## Dataset Download

HM3D dataset download is completed using the steps outlined under "Downloading the HM3D dataset" in VLFM's README: https://github.com/rai-opensource/vlfm#dart-downloading-the-hm3d-dataset

1. **Obtain a Matterport Token ID and Secret**

2. **Set environment variables**
   ```bash
   export MATTERPORT_TOKEN_ID=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
   export MATTERPORT_TOKEN_SECRET=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
   export DATA_DIR=</path/to/data> # e.g. /home/mak/research/datasets/hm3d/data
   export HM3D_OBJECTNAV=https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
   ```

   We recommend creating an external directory to house the dataset, and creating symbolic links to the dataset directories inside the RPV `data` directory.

3. **Download the HM3D validation set**
   ```bash
   python -m habitat_sim.utils.datasets_download \
     --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
     --uids hm3d_val_v0.2 \
     --data-path $DATA_DIR &&

   # Download HM3D ObjectNav dataset episodes
   wget $HM3D_OBJECTNAV &&
   unzip objectnav_hm3d_v1.zip &&
   mkdir -p $DATA_DIR/datasets/objectnav/hm3d &&
   mv objectnav_hm3d_v1 $DATA_DIR/datasets/objectnav/hm3d/v1 &&
   rm objectnav_hm3d_v1.zip
   ```

   **Note:** These steps download HM3DSem-v0.2 scenes paired with `objectnav_hm3d_v1.zip` episodes. Per habitat-lab's dataset documentation, v1 episodes are officially paired with v0.1 scenes — however, this is the exact combination specified in VLFM's README, and is what was used to produce the results in this repository.

   Verify with `find -maxdepth 4` that the directory tree follows the structure:
   ```
   .
   ./scene_datasets
   ./scene_datasets/hm3d
   ./versioned_data
   ./versioned_data/hm3d-0.2
   ./versioned_data/hm3d-0.2/val-habitat-files.json.gz
   ./versioned_data/hm3d-0.2/val-semantic-configs-files.json.gz
   ./versioned_data/hm3d-0.2/val-configs-files.json.gz
   ./versioned_data/hm3d-0.2/val-semantic-annots-files.json.gz
   ./versioned_data/hm3d-0.2/hm3d
   ./versioned_data/hm3d-0.2/hm3d/val
   ./versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json
   ./datasets
   ./datasets/objectnav
   ./datasets/objectnav/hm3d
   ./datasets/objectnav/hm3d/v1
   ```

4. **Copy `hm3d_annotated_val_basis` into `scene_datasets/hm3d`**
   ```bash
   cd <path to ${DATA_DIR}/versioned_data/hm3d-0.2/hm3d/val>
   cp hm3d_annotated_val_basis.scene_dataset_config.json ..
   ```

5. **Create symbolic links from the dataset installation location to the RPV-SemNav `data` directory**
   ```bash
   cd <path to ${RPV_ROOT}/data>
   ln -s <path to ${DATA_DIR}/datasets> datasets
   ln -s <path to ${DATA_DIR}/scene_datasets> scene_datasets
   ```

## Model Checkpoints

Required checkpoints:
- `sam3.pt`
- `yoloe-26x-seg.pt`
- `ade20k-semseg-r50_model_final_500878.pkl`

Download links for these checkpoints are a work in progress. Once obtained, checkpoints should be saved to the `checkpoints` directory inside the project root directory.

## Running Evaluation

Running evaluation requires two terminal windows: one to launch the models for the vision pipeline, and one to run the Habitat simulator.

1. **Launch the vision pipeline models**

   From the project root directory:
   ```bash
   ./scripts/launch_dl_servers.sh
   ```
   Note: you may need to run `chmod +x` on this file first.

2. **Run the evaluation script**

   Run the following to evaluate on the HM3D dataset:
   ```bash
   python -m vlfm.run
   ```

   To save video output of validation episodes, run with the following environment variables set:
   ```bash
   python -m vlfm.run habitat_baselines.video_dir=${VIDEO_DIR} \
     habitat_baselines.eval.video_option=${VIDEO_OPTION} \
     habitat_baselines.video_fps=${VIDEO_FPS}
   ```

   For example, we set:
   ```bash
   VIDEO_DIR=<path to directory to save video output>
   VIDEO_OPTION='["disk"]'
   VIDEO_FPS=2
   ```

   To save episode data to a CSV file at the end of the evaluation, you can also set the `CSV_PATH` environment variable. This is not required to be passed as an argument. For example:
   ```bash
   export CSV_PATH={path to logging directory}/eval_stats.csv
   ```
