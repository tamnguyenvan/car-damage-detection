# Model Training Guide

This directory contains resources to train and fine-tune Ultralytics YOLO object-detection models for car damage detection. You have two primary options for training.

## Prerequisites
*   **No local GPU required**: Both training methods execute entirely in the cloud.
*   **Automatic Data Provisioning**: The 6GB CarDD dataset is hosted on Google Drive. Both the Colab notebook and the Modal script will download and format this dataset automatically. You do not need to download it manually.
*   **Accounts**: You will need a Modal account (for Option 1) or a free Google account (for Option 2).

## Option 1: Modal (Serverless & Automated) - ⭐ Recommended

If you prefer an automated, hands-off cloud training pipeline, you can use the `modal_train.py` script. We highly recommend this option because it leverages [Modal](https://modal.com) to automatically provision much more powerful cloud GPUs (like the H100 or A100) to significantly speed up training. The script will dynamically download and process the dataset, train the model, and persist the results entirely in the cloud.

### Setup
1. Create an account on Modal.
2. Install the Modal client locally:
   ```bash
   # Create virtual env
   python3.11 -m venv venv_modal
   source venv_modal/bin/activate

   pip install modal
   ```
3. Authenticate your CLI:
   ```bash
   modal setup
   ```

### Run the Training Job
Execute the following command from the root of the project:
```bash
cd notebooks
# Run with the default object-detection model (YOLOv8m)
modal run modal_train.py

# Or specify a different YOLO detection model:
modal run modal_train.py --model-name yolo11m.pt

# Optionally set the seed for reproducibility:
modal run modal_train.py --model-name yolov8m.pt --seed 42
```

> [!NOTE]
> The first time you run this, Modal will build the container image. During the image build phase, it will download the 6GB dataset and convert COCO bounding boxes to YOLO object-detection labels. This will take ~5-10 minutes. Subsequent runs will be nearly instantaneous as the dataset is permanently "baked" into the image.

### Retrieve Your Model
The training script saves the output to a persistent Modal Volume named `car-damage-training-vol`. To download your newly trained weights to your local machine, run:
```bash
modal volume get car-damage-training-vol /car_damage_finetuned/weights/best.pt ./models/best.pt
```

## Option 2: Google Colab (Interactive)

The interactive Jupyter Notebook (`training.ipynb`) provides a step-by-step workflow for fine-tuning the model using a free GPU on Google Colab.

1. Go to [Google Colab](https://colab.research.google.com/).
2. Upload the `training.ipynb` file from this directory.
3. In Colab, go to **Runtime > Change runtime type** and select a **T4 GPU** or **V100/A100 GPU** if available.
4. Run all the cells in the notebook.
5. Once training completes, you can download the final `best.pt` weights and place it in the `models/` directory of your project.
