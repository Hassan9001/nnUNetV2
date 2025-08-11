# nnUNetV2

# pancreas-mtl-nnunet2d

2D multi-task **pancreas/lesion segmentation** + **subtype classification** using **nnU-Net v2** on Windows + Anaconda (no Docker).

> Tested with: Python 3.10, PyTorch (CUDA 12.1 wheels), nnU-Net v2 (editable install), NVIDIA RTX 3080 (10–12 GB).

## 1) Install (Windows + Anaconda)

```bat
conda info # Optional: info about current conda install.
conda create -y -n nnunetTEST python=3.10
conda info --envs # Optional: to list conda enviorments available
conda activate nnunetTEST

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install SimpleITK nibabel pandas numpy matplotlib

git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
cd ..

git clone <THIS_REPO_URL> pancreas-mtl-nnunet2d
cd pancreas-mtl-nnunet2d
pip install -e .

REM set paths (open a NEW terminal after setx or set temporarily with `set`)
setx nnUNet_raw "C:\nnUNet_raw"
setx nnUNet_preprocessed "C:\nnUNet_preprocessed"
setx nnUNet_results "C:\nnUNet_results"
```

## 2) Prepare data

Place your original folder as `./sourcedata/` (see report for exact structure). Then run conversion:

```bat
python scripts\convert_to_nnunetv2.py ^
  --src .\sourcedata ^
  --dst %nnUNet_raw% ^
  --dataset_id 777
```

This creates `%nnUNet_raw%\Dataset777_PancreasSegCls` with `imagesTr`, `labelsTr`, `imagesTs`.

## 3) Plan & preprocess (2D)

```bat
nnUNetv2_plan_and_preprocess -d 777 -c 2d --verify_dataset_integrity
```

## 4) Train 5 folds

```bat
nnUNetv2_train 777 2d 0 -tr nnunet_ext.trainers.trainer_segcls2d.Trainer_SegCls2D --npz
nnUNetv2_train 777 2d 1 -tr nnunet_ext.trainers.trainer_segcls2d.Trainer_SegCls2D --npz
nnUNetv2_train 777 2d 2 -tr nnunet_ext.trainers.trainer_segcls2d.Trainer_SegCls2D --npz
nnUNetv2_train 777 2d 3 -tr nnunet_ext.trainers.trainer_segcls2d.Trainer_SegCls2D --npz
nnUNetv2_train 777 2d 4 -tr nnunet_ext.trainers.trainer_segcls2d.Trainer_SegCls2D --npz
```

## 5) Predict (fast preset, ≥10% faster)

```bat
python nnunet_ext\inference\predict_segcls_2d.py ^
  --dataset_id 777 ^
  --folds 0 1 2 3 4 ^
  --input %nnUNet_raw%\Dataset777_PancreasSegCls\imagesTs ^
  --output .\your_name_results ^
  --speed_cfg .\configs\speed.yaml
```

Outputs: `quiz_XXX.nii.gz` masks + `subtype_results.csv (Names,Subtype)`.

## 6) Evaluate on original validation (optional)

```bat
python scripts\evaluate_val.py ^
  --val_images %nnUNet_raw%\Dataset777_PancreasSegCls\imagesTr_from_validation ^
  --val_labels .\sourcedata\validation ^
  --pred_dir .\your_name_results_val
```

## 7) Package submission

```bat
python scripts\pack_submission.py ^
  --pred_dir .\your_name_results ^
  --out_zip  .\your_name_results.zip
```

For full methodology, diagrams, and fixed hyperparameters, see `reports\Pancreas_MTL_Report.md`.

---

## Dependencies & Requirements

To create a virtual environment and install the required dependences please follow steps (w/ Anaconda):

- Open fresh new VScode window
- Select Your Conda Environment in VSCode:
  - Open the Command Palette in VSCode: `Ctrl + Shift + P`
  - Type and select: `Python: Select Interpreter`
  - Pick your Conda environment (Ex: base (3.12.7) ~\anaconda3\python.exe)
- Use the Terminal in VSCode with Conda
  - Open new terminal: `Ctrl + shift + ~`
  - Clone repository: run the following (in your working folder):

    ```shell
    cd Desktop/code # Optional: Nav to dif dir (where you want the repo to save) 
    git clone https://github.com/mie-lab/location-mode-prediction
    cd location-mode-prediction
    ```

  - Make Virtual Enviorment & Download dependencies: run the following on Git Bash Terminal:

    ```shell
    conda info # Optional: info about current conda install.
    conda env create -f environment.yml #OR# conda create -y -n loc-mode-pred python=3.10
    conda info --envs # to list conda enviorments available
    conda activate loc-mode-pred
    ```

  - If redo needed, can run:

    ```shell
    conda deactivate
    conda env remove -n loc-mode-pred
    conda env create -f environment.yml
    conda activate loc-mode-pred
    ```

  - other ex:

    ```shell
    # downgrade numpy
    pip install "numpy<2" #numpy==1.26.4

    # fresh install with old numpy compiler
    pip uninstall shapely geopandas -y
    pip install shapely==2.0.7 geopandas==0.12.2 #0.10.2
    #+ this downgrade
    pip install pandas==1.5.3 trackintel==1.1.13 #1.2.1

    # check
    python -c "import numpy, shapely, geopandas; print(numpy.__version__, shapely.__version__, geopandas.__version__)"# output:: 1.26.4 2.0.1 0.14.3

    # create requirements.txt file
    pip freeze > requirements.txt #or
    pip list --format=freeze > requirements.txt #this
    ```
