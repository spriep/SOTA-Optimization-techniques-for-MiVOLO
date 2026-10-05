# Age & Gender Estimation: Optimization, Quantization and Distillation Pipeline

This project focuses on the development of computer vision pipelines, optimization, and model compression applied to the multi-task architecture **MiVOLO-D1** for real-time age estimation and gender classification (evaluated strictly in a face-only configuration) on the **UTKFace** and **MORPH II** benchmark datasets.

## 💻 Hardware and Software Experimental Setup

All experiments, benchmarking iterations, and training pipelines reported in this study were executed under the following local hardware and software specifications:

* **Operating System:** Windows 11
* **CPU:** AMD Ryzen 7 5800H with Radeon Graphics
* **System RAM:** 16 GB
* **Integrated GPU:** AMD Radeon Graphics
* **Discrete GPU:** NVIDIA GeForce RTX 3050

## 2 The Baseline Architecture: MiVOLO-D1

Although the full MiVOLO framework allows simultaneous processing of face and body crops, the experimental scope of this research was strictly limited to facial images (*face-only*). The inputs are normalized and resized to \(224 x 224\) pixels, producing the estimated continuous age, gender confidence score, and binary gender classification as outputs.

### 2.1 Selection of the Primary Baseline Model

To establish the starting point before applying compression techniques, the official pre-trained weights available in the MiVOLO repository were evaluated using FP32 precision and single-image inference (batch size = 1).

Based on the comparative performance evaluations, `model_imdb_age_gender_4.22.pth.tar` was selected as the definitive baseline checkpoint for all subsequent optimization, quantization (PTQ FP16/INT8), and knowledge distillation experiments.

| Model | Dataset | MAE (years) | RMSE | Gender Accuracy (%) | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: |
| model_imdb_age_gender_4.22.pth.tar | UTKFace | 5.33 | 7.18 | 92.13 | 41.46 |
| model_imdb_age_gender_4.22.pth.tar | MORPH2 | 4.46 | 5.60 | 97.28 | 41.84 |


This model has a baseline disk size of **98.70 MB** and a peak VRAM consumption of **115.77 MB** during execution.


## 3 Benchmark Datasets and Preprocessing Pipeline

The datasets that were used:

* **UTKFace:** A facial dataset featuring wide variability in age, ethnicity, and gender. A total of 23,708 valid images.
* **MORPH2:** A large-scale longitudinal facial database commonly used for age progression and demographic estimation.

### 3.1 MORPH2 Face Preprocessing Pipeline via RetinaFace

Given that MiVOLO-D1 was strictly evaluated in a *face-only* mode, an automated facial extraction step was applied to the 55,608 images in the MORPH2 dataset using RetinaFace. During this preprocessing stage, 676 images were discarded, resulting in a dataset of 54,932 images.

### 3.2 Performance Metrics and Evaluation Protocol

Tests were designed using an execution protocol with a batch size of 1 on CUDA to reflect real-world deployment constraints on edge devices, with results averaged across multiple iterations.

* **Age Regression Metrics:**
  * **Mean Absolute Error (MAE)**
  * **Root Mean Squared Error (RMSE)**
* **Gender Classification Metrics:**
  * **Gender Accuracy (%)**
* **Inference Performance:**
  * **Frames Per Second (FPS)**
  * **Total Inference Time**
* **Hardware and Storage Consumption:**
  * **Peak VRAM (MB):** Maximum video memory allocation required on the GPU during execution. 
  * **Model Size (MB):** Physical disk storage space occupied by the model file.


## 4 Post-Training Quantization in FP16 (PTQ FP16)

From an implementation perspective, the MiVOLO base model offers native FP16 support within its pipeline, allowing weights to be converted and loaded in half-precision directly onto the CUDA device upon instantiating the estimator.

Precise results can be found in the project report.


### 4.2 Performance Analysis and Findings

* **Memory and storage footprint:** Conversion to half-precision achieved an exact **50.0% reduction in on-disk model size**, dropping from 98.67 MB to 49.33 MB. Similarly, peak video memory consumption during **inference decreased by 44.73%**, falling from 115.77 MB to 63.98 MB.
* **Performance and inference:** On the UTKFace dataset, the processing rate increased from 41.46 to 45.41 FPS, **reducing total time by 8.70%**. On MORPH II, performance reached 44.60 FPS (compared to 36.55 FPS in FP32), representing a **19.72% reduction in overall evaluation time**.
* **Accuracy preservation and improvement:** On UTKFace, the **MAE remained unchanged at 5.33 years**. On MORPH II, **the MAE dropped** from 4.86 to 4.46 years, and gender accuracy rose from 97.10% to 97.29%. This slight gain is attributed to the implicit regularization effect provided by FP16 numerical rounding, which mitigates sensitivity to the high-frequency noise associated with overfitting in FP32.

## 5 Post-Training Quantization in INT8

Following the evaluation of FP16 precision, 8-bit integer Post-Training Quantization (PTQ INT8) was explored to achieve higher memory compression and throughput, encountering several technical constraints:

### 5.1 Native CUDA Limitations
Native execution under PyTorch on CUDA presents structural limitations: backends supporting INT8 computation (such as FBGEMM and QNNPACK) are explicitly engineered and optimized for x86 and ARM CPU architectures. Standard PyTorch execution lacks native INT8 compute kernels for CUDA. Consequently, although model weights can be physically stored in INT8 precision, they are dynamically dequantized back to FP32 or FP16 during inference immediately before matrix operations are executed. Rather than accelerating throughput, this continuous runtime dequantization introduces computational overhead that ultimately results in slower end-to-end inference compared to native floating-point execution.

### 5.2 Compilation Incompatibilities in ONNX and TensorRT
To bypass runtime dequantization penalties and unlock dedicated 8-bit acceleration kernels, industry deployment workflows rely on static inference compilers such as NVIDIA TensorRT, which require a static computation graph via ONNX export. When attempting to export MiVOLO-D1, this compilation pipeline failed due to two architectural characteristics:
* **Dynamic Execution and State Modification:** Built on top of the `timm` library, MiVOLO relies on custom modules that dynamically modify internal states during runtime, preventing the construction of a fixed, static computation graph.
* **Unsupported Graph Operators:** The export process fails on dynamic spatial tensor operations embedded within the model—most notably `fold`, `unfold`, and `col2im` operations present in custom layers—which modern deployment backends like TensorRT and Intel OpenVINO cannot parse or compile.

### 5.3 Evaluation of Alternative PyTorch Quantization Frameworks
Three PyTorch-centric libraries were systematically evaluated to explore INT8 quantization without relying on static ONNX or TensorRT export pipelines:
* **Optimum.quanto:** Replaces standard parameters with custom quantized types (`QBytes`), triggering persistent type mismatch exceptions with internal `timm` components that expect standard PyTorch floating-point tensors.
* **torch.ao:** Discarded due to the lack of direct CUDA execution support for standard INT8 quantization workflows.
* **torchao.quantization:** Successfully converted layers into affine-quantized structures (`AffineQuantizedTensor`) without runtime exceptions. However, execution profiling confirmed that internal MiVOLO methods continued to recast quantized tensors back to FP32 prior to GPU kernel execution, preserving the exact runtime dequantization bottleneck observed in native PyTorch.

## 6 Knowledge Distillation

Considering the limitations encountered in the different attempts to implement INT8 PTQ, Knowledge Distillation (KD) was adopted. This approach uses the full-precision MiVOLO-D1 model as a static Teacher to supervise compact, exportable Student networks adapted for inference hardware.

### 6.1 Student Architectures

The Student networks were integrated through a unified wrapper class (`StaticStudentMiVOLO`) using the timm library. Four representative architectures were selected after evaluating more than twenty candidate models, considering their functional compatibility with MobileNetV3:
* **Simple_vits (`vit_tiny_patch16`):** Lightweight variant of the classic Vision Transformer.
* **MobileNetV3-Large (`mobilenetv3_large_100`):** Convolutional architecture optimized with depthwise separable blocks.
* **MobileViT (`mobilevit_xs`):** Hybrid model combining convolutional layers with self-attention mechanisms from visual transformers.
* **GhostNet (`ghostnet_100`):** Convolutional network based on efficient feature map generation through Ghost blocks.

### 6.2 Data Organization and Logit Caching

A standard 80% training and 20% test split was implemented for the datasets. To avoid the computational redundancy of querying the Teacher at every epoch, a preprocessing extraction pipeline (`extract_teacher_logits.py`) was implemented. The continuous, unnormalized logits produced by the frozen Teacher were precomputed and serialized to disk within a persistent directory (`teacher_logits_cache`) as PyTorch tensor files (.pt). The custom dataset class (`Morph2UTKDataset`) loaded the images together with their cached logits in a decoupled manner, eliminating VRAM bottlenecks and accelerating training.

### 6.3 Two-Phase Training Protocol

To overcome the representation gap between the Student architectures and the Teacher, training was structured into two stages:
* **Phase 1: Gender-based Student preconditioning (1-2 epochs):** Initial supervised optimization applied exclusively to the gender classification head using Cross-Entropy loss with respect to the ground-truth labels, allowing the Student to learn stable facial features.
* **Phase 2: Supervised multitask knowledge distillation (3-6 epochs):** Joint optimization for age estimation (MSE) and gender classification (Cross-Entropy). A Teacher-Bounded scheme was integrated (\($\alpha=0.5$\) and temperature \(T=1\)), in which Teacher logit supervision is activated only for individual samples where the Teacher's error is strictly lower than the Student's error with respect to the ground-truth label.

### 6.4 Performance Results

Precise results can be found in the project report.

### 6.5 Performance Analysis and Key Findings

* **Hardware efficiency and compression:** **All architectures studied reduced weight storage** to a range between 7.62 MB and 21.19 MB (a reduction of 78.5% to 92.3% compared with the Teacher's 98.70 MB). Peak video memory remained below 33 MB, **reducing consumption by approximately 72%** compared with the baseline.
* **Speed-error trade-off:** MobileNetV3-Large provided the most balanced behavior for production deployment, approximately **doubling the processing rate** to 77.60-82.12 FPS (compared with 40.6 FPS for the baseline) while maintaining low degradation in age regression.
* **Outperforming the Baseline in-domain:** GhostNet distilled directly on MORPH2 **outperformed the original baseline model** in in-domain accuracy, achieving an MAE of 4.22 years (compared with 4.45 for the Teacher), an RMSE of 5.39, and a gender classification accuracy of 99.28%.
* **Cross-dataset generalization boundaries:** Students distilled on UTKFace retained predictive capability when evaluated on MORPH2 (MAE of approximately 4.45-4.46 years). In contrast, models trained on MORPH2 experienced substantial degradation when transferred to UTKFace (MAE of 14.09-15.77 years). This disparity is attributed to the demographic heterogeneity of UTKFace (age range from 0 to 100 years with fewer environmental constraints) compared with the narrower age range (16 to 77 years) and more homogeneous capture conditions of MORPH2.

## 7 Future Work

1. **Exploration of additional optimization techniques:** Beyond the post-training quantization (PTQ) and logit-based distillation approaches analyzed in this project, future work will investigate and benchmark complementary techniques described in the theoretical framework, such as structured 2:4 pruning with Tensor Core support and parameter-efficient fine-tuning (PEFT).
2. **Expansion of knowledge distillation training budgets:** The distillation experiments were conducted under limited training schedules. A systematic exploration using a larger number of epochs, decay schedules, and adaptive learning-rate schedulers will help determine whether lightweight architectures can achieve higher regression fidelity without overfitting the supervision split.
3. **Refinement of the loss function with adjustable temperature:** The current loss function operates directly on the teacher's raw logits with per-instance reductions (\(T=1\)). Incorporating a configurable temperature parameter (\(T\)) into the gender classification loss represents a key adjustment for controlling the transfer of soft probability distributions.
4. **Transition to Dual-Crop modality (MiVOLO-D2):** The scope of this study was strictly limited to facial processing with MiVOLO-D1, excluding full-body person detection. Future work should extend the pipeline to the MiVOLO-D2 architecture, simultaneously processing face and full-body crops. Evaluating this dual modality will make it possible to measure the extent to which complementary contextual information improves demographic estimation and to determine whether these full-body features can be effectively transferred to compact student models.

## 8 Execution Instructions

### 8.1 Knowledge Distillation

1. **Extract the Teacher logits:** Run the logits extraction script provided in the repository. This generates the cached Teacher outputs required during Student training.
2. **Set the Teacher logits path:** In the `main` section of the distillation script, update the path to point to the directory where the extracted Teacher logits are stored.
3. **Set the training dataset path:** In the same script, configure the path to the training dataset that will be used for Student training.
4. **Select the Student backbone:** In the `StudentObject` file, modify the backbone configuration to select the architecture that will be used as the Student model.
5. **Run the distillation:** Once the paths and backbone have been configured, execute the distillation script to train the selected Student architecture using the cached Teacher logits.

### 8.2 Benchmark Execution


1. **Select the backbone:** In the `StudentObject` file, change the backbone to the architecture that is going to be benchmarked.
2. **Set the model weights:** Configure the path to the weights of the trained model that will be evaluated.
3. **Select the estimator:** Specify which estimator class should be used for the selected model.
4. **Select the dataset:** Change the `CURRENT_DS` variable according to the dataset being evaluated (utk/morph).
5. **Set the dataset paths:** Configure the corresponding paths to the UTKFace or MORPH2 dataset.
6. **Run the benchmark:** Execute the benchmark pipeline with the selected model, weights, estimator, and dataset configuration.

### 8.3 Dynamic FP16 Execution

To execute the model using Dynamic FP16, no additional pipeline configuration is required.

Simply open the `BasicPipeline` configuration and modify the `quant_type` variable to select the FP16 quantization mode.

After changing `quant_type`, run the pipeline normally to obtain the Dynamic FP16 benchmark results.

