# Vision-to-Physics (V2P) framework

This repository contains the reproducibility package for the **Vision-to-Physics (V2P)** framework, which bridges generative occupant modeling with building thermodynamics. Inspired by the [BEDLAM](https://bedlam.is.tue.mpg.de/) dataset, this framework leverages [SMPL-X](https://smpl-x.is.tue.mpg.de/) body models and [AMASS motion sequences](https://amass.is.tue.mpg.de/) within Unreal Engine 5 to render highly realistic synthetic occupant videos, while capturing the ground truth physical vectors (Metabolic Rate (MET) and Clothing Insulation (CLO), and 3D spatial coordinates). 

Additionally, this repository provides the evaluation scripts and prompt pipelines used to benchmark zero-shot Vision-Language Models (VLMs) against supervised Convolutional Neural Networks (CNNs) for extracting physics-based thermodynamic metrics (MET and CLO) from these visual inputs.

## Repository Structure

* **`/Sample_Data:** Contains representative synthetic video clips generated from the UE5 pipeline, alongside their corresponding ground truth JSON vectors.
  * `/Videos`: Representative `.mp4` video clips.
  * `/Vectors`: matching `.json` files containing the evaluation vectors, including 3D spatial coordinates.
* **`/Scripts:** Contains the core logic required to reproduce the VLM evaluation pipeline and the CNN cross-validation splits.
  * `annotate_clip_gemini_updated.py`: Contains the exact chain-of-thought prompt templates, JSON schemas, and ASHRAE mapping dictionaries used for the Gemini API.
  * `batch_annotate_gemini.py`: The batch execution script for processing multiple videos.
  * `preprocess_data.py`: Prepares the raw JSON outputs into standardized numeric arrays for evaluation.
  * `evaluate_activity_metrics.py`: Calculates Top-1 Accuracy, Top-5 Accuracy, and MET errors (MAE/RMSE).
  * `evaluate_clothing_metrics.py`: Calculates multi-label Intersection over Union (IoU) and CLO errors (MAE/RMSE).
  * `split_identifiers.json`: Documents the exact clip IDs distributed across the 5 folds used during the supervised CNN cross-validation.
* `requirements.txt`: List of Python dependencies required to run the evaluation scripts.
