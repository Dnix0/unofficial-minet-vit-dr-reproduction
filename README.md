# MINet-ViT 당뇨망막병증 등급 분류 비공식 재현

> **프로젝트 상태:** 본 저장소는 원 논문의 공식 구현이 아닌, 논문에 공개된 방법론을 바탕으로 독립적으로 작성한 비공식 부분 재현 코드입니다. 원 저자, Wiley 또는 해당 학술지의 공식 저장소가 아닙니다.

## 1. 원 논문

- **제목:** A Novel Noise Removal and Interpretable Deep Learning Model for Diabetic Retinopathy Detection
- **저자:** Sultan Alanazi, Sajid Ullah Khan, Faisal M. Alotaibi, Mohammed Alonazi
- **학술지:** International Journal of Imaging Systems and Technology
- **연도 및 논문번호:** 2025, 35:e70245
- **DOI:** https://doi.org/10.1002/ima.70245

## 2. 프로젝트 목적

본 프로젝트의 목적은 공식 소스 코드가 제공되지 않은 상황에서 논문에 기술된 핵심 아이디어를 이해하고, APTOS 2019 데이터셋을 이용해 재현 가능한 형태로 독립 구현하는 것입니다.

논문의 완전한 구현 세부사항과 공식 학습 코드가 공개되지 않았으므로, 본 저장소는 논문 성능을 그대로 복제한 공식 재현이 아니라 다음 핵심 요소를 중심으로 수행한 **구조 기반 부분 재현**입니다.

- 안저 영상의 노이즈 저감
- CNN의 국소 특징과 Vision Transformer의 전역 특징 결합
- 5단계 당뇨망막병증 등급 분류
- Grad-CAM 기반 정성적 설명 가능성 분석
- 전처리 및 denoising 구성요소에 대한 ablation study

## 3. 재현 범위

### 구현한 내용

- APTOS 2019 기반 5-class DR grading
- Label-stratified image-level 70:10:20 split
- EfficientNet-B0 baseline
- ResNet18 CNN branch와 lightweight ViT branch를 결합한 Hybrid CNN-ViT
- Gaussian/Poisson synthetic noise 기반 Residual Denoising U-Net 사전학습
- 원본 영상과 denoised 영상의 alpha blending
- VST/Retinex strong 및 mild exploratory ablation
- Accuracy, Macro F1, Weighted F1, QWK, confusion matrix 평가
- Grad-CAM 시각화

### 정확히 재현하지 못한 내용

- 원 논문의 Multi-level Imperialistic U-Net, MINet 전체 구조
- Imperialistic Learning Optimization, ILO의 전체 최적화 절차
- 원 저자가 사용한 정확한 이미지 ID 목록과 random seed
- SHAP 분석
- EyePACS 실험

## 4. 주요 구현 차이

논문의 MINet 세부 구현과 공식 코드가 제공되지 않아, 본 프로젝트에서는 Residual Denoising U-Net을 대체 노이즈 제거 모듈로 사용했습니다.

또한 다음 구성은 논문의 공식 구현으로 확인된 요소가 아니라 본 재현 과정에서 추가한 탐색적 실험입니다.

- Retinex 및 gamma 기반 조명 보정
- Green-channel guidance
- 원본 영상과 denoised 영상의 alpha blending

따라서 본 저장소의 결과는 원 논문의 공식 결과가 아니며, 논문의 핵심 아이디어를 독립적으로 구현한 결과로 해석해야 합니다.

## 5. 데이터셋과 분할

본 저장소는 APTOS 2019 또는 EyePACS의 원본 이미지, label 파일, CSV manifest 및 압축파일을 배포하지 않습니다. 데이터는 정식 제공처에서 직접 내려받고 해당 이용 조건을 따라야 합니다.

최종 재현 실험에서는 APTOS 2019의 3,662개 이미지를 다음과 같이 분할했습니다.

| Split | 이미지 수 | 비율 | 용도 |
|---|---:|---:|---|
| Train | 2,563 | 70% | 모델 학습 |
| Validation | 366 | 10% | Validation QWK 기반 checkpoint 선택 |
| Test | 733 | 20% | 최종 성능 평가 |

본 프로젝트의 분할은 **label-stratified image-level split**입니다. 논문에 보고된 분할 수와 동일하지만, 원 논문의 정확한 이미지 목록과 random seed가 공개되지 않았으므로 동일한 sample split이라고 주장하지 않습니다. 또한 patient ID가 없으므로 patient-level split으로 검증되지 않았습니다.

자세한 데이터 준비 방법은 [`DATASET.md`](DATASET.md)를 참고하세요.

## 6. 최종 재현 결과

아래 결과는 현재 저장되었던 final ep20 checkpoint를 동일한 test pipeline으로 다시 평가해 얻은 본 프로젝트의 결과이며, 원 논문의 공식 결과가 아닙니다.

| 모델 | Accuracy | Macro F1 | Weighted F1 | QWK |
|---|---:|---:|---:|---:|
| Hybrid CNN-ViT ep20 | 0.7872 | 0.6284 | 0.7886 | 0.8665 |
| Denoised Hybrid, alpha=0.25 ep20 | 0.7776 | 0.6296 | 0.7831 | 0.8747 |
| **Denoised Hybrid, alpha=0.5 ep20** | **0.8022** | **0.6423** | **0.8021** | **0.8955** |

현재 구현한 모델 중에서는 Denoised Hybrid alpha=0.5 ep20이 가장 높은 성능을 기록했습니다.

## 7. 설치

Python 환경을 생성한 뒤 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

GPU 학습에는 운영체제와 CUDA 버전에 맞는 PyTorch 설치가 필요할 수 있습니다. PyTorch와 torchvision은 사용하는 CUDA 환경에 맞춰 설치하는 것을 권장합니다.

## 8. 로컬 데이터 경로 설정

데이터와 split CSV는 저장소에 포함되지 않습니다. 학습 또는 평가 전에 환경변수로 로컬 manifest 경로를 지정할 수 있습니다.

### Windows PowerShell

```powershell
$env:APTOS_TRAIN_CSV = "D:\your-local-path\train.csv"
$env:APTOS_VAL_CSV = "D:\your-local-path\val.csv"
$env:APTOS_TEST_CSV = "D:\your-local-path\test.csv"
```

### Windows Command Prompt

```bat
set APTOS_TRAIN_CSV=D:\your-local-path\train.csv
set APTOS_VAL_CSV=D:\your-local-path\val.csv
set APTOS_TEST_CSV=D:\your-local-path\test.csv
```

환경변수가 설정되지 않은 경우 스크립트는 다음 상대경로를 기본값으로 사용합니다.

```text
data/aptos2019_processed/train.csv
data/aptos2019_processed/val.csv
data/aptos2019_processed/test.csv
```

로컬 `data/` 폴더는 `.gitignore`에서 제외됩니다.

## 9. 주요 실행 순서

저장소의 루트 폴더에서 실행합니다.

### Baseline 학습

```bash
python train_baseline.py
```

### Hybrid CNN-ViT 학습

```bash
python train_hybrid_cnn_vit_ep20.py
```

### Residual Denoising U-Net 학습

```bash
python train_denoiser.py
```

### 최종 Denoised Hybrid 학습

```bash
python train_denoised_hybrid_cnn_vit_alpha05_ep20_timed.py
```

### 최종 모델 평가

```bash
python test_final_ep20_models.py
```

### Grad-CAM 대상 선정 및 생성

```bash
python select_gradcam_cases.py
python run_gradcam_final_ep20.py
```

Checkpoint, prediction CSV, 의료영상 및 생성된 output은 본 저장소에 포함되지 않습니다.

## 10. 선택적 VST/Retinex 전처리

전처리 스크립트는 다음 환경변수를 지원합니다.

```powershell
$env:APTOS_PROCESSED_DIR = "D:\your-local-processed-data"
$env:APTOS_VST_RETINEX_DIR = "D:\your-output\vst-retinex"
$env:APTOS_VST_RETINEX_MILD_DIR = "D:\your-output\vst-retinex-mild"
```

실행:

```bash
python prepare_vst_retinex_dataset.py
python prepare_vst_retinex_mild_dataset.py
```

VST/Retinex 실험은 탐색적 ablation이며, 원 논문의 MINet 전처리를 정확히 구현한 것으로 해석하면 안 됩니다.

## 11. 인용

본 저장소에서 구현한 과학적 아이디어를 사용하는 경우 원 논문을 우선 인용해 주세요.

```text
Alanazi, S., Khan, S. U., Alotaibi, F. M., & Alonazi, M. (2025).
A Novel Noise Removal and Interpretable Deep Learning Model for
Diabetic Retinopathy Detection.
International Journal of Imaging Systems and Technology, 35, e70245.
https://doi.org/10.1002/ima.70245
```

## 12. 라이선스와 제3자 코드

현재 소스 코드 라이선스는 최종 결정 전입니다. 저장소를 공개로 전환하기 전에 코드 소유권과 공개 범위를 지도교수 또는 소속기관과 확인해야 합니다.

외부 패키지와 제3자 코드 또는 모델 구성요소는 각 원본 라이선스의 적용을 받습니다. 관련 내용은 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)에 정리합니다.

---

# Unofficial MINet-ViT Reproduction for Diabetic Retinopathy Grading

> **Project status:** This is an independent, unofficial, and partial reproduction. It is not an official implementation by the original authors and is not affiliated with or endorsed by Wiley or the journal.

## 1. Original paper

- **Title:** A Novel Noise Removal and Interpretable Deep Learning Model for Diabetic Retinopathy Detection
- **Authors:** Sultan Alanazi, Sajid Ullah Khan, Faisal M. Alotaibi, Mohammed Alonazi
- **Journal:** International Journal of Imaging Systems and Technology
- **Year and article number:** 2025, 35:e70245
- **DOI:** https://doi.org/10.1002/ima.70245

## 2. Purpose

This repository documents an independent attempt to reproduce the core ideas described in the paper using APTOS 2019. Because official source code and several implementation details were unavailable, this project should be understood as a structural and conceptual partial reproduction rather than an exact reproduction of the original results.

The main reproduced concepts are:

- retinal fundus image denoising,
- fusion of local CNN features and global Vision Transformer features,
- five-class diabetic retinopathy grading,
- Grad-CAM-based qualitative interpretation, and
- ablation studies for preprocessing and denoising components.

## 3. Reproduction scope

### Implemented

- APTOS 2019 five-class DR grading
- Label-stratified image-level 70:10:20 split
- EfficientNet-B0 baseline
- Hybrid CNN-ViT using a ResNet18 CNN branch and a lightweight ViT branch
- Residual denoising U-Net pretrained with synthetic Gaussian and Poisson noise
- Original-denoised image alpha blending
- Strong and mild VST/Retinex exploratory ablations
- Accuracy, Macro F1, Weighted F1, QWK, and confusion-matrix evaluation
- Grad-CAM visualization

### Not reproduced exactly

- The complete Multi-level Imperialistic U-Net architecture
- The complete Imperialistic Learning Optimization procedure
- The exact image identifiers and random seed used by the authors
- SHAP analysis
- EyePACS experiments

## 4. Important implementation differences

The original MINet was replaced with a residual denoising U-Net because complete implementation details and official code were unavailable.

Retinex processing, gamma adjustment, green-channel guidance, and original-denoised alpha blending were evaluated as additional exploratory strategies in this reproduction. They must not be interpreted as confirmed components of the original authors' official implementation.

## 5. Dataset and split

This repository does not distribute APTOS 2019 or EyePACS images, labels, CSV manifests, or archives. Users must obtain the datasets from their authorized sources and comply with the applicable terms.

| Split | Images | Ratio | Purpose |
|---|---:|---:|---|
| Train | 2,563 | 70% | Model training |
| Validation | 366 | 10% | Checkpoint selection using validation QWK |
| Test | 733 | 20% | Final evaluation |

The split is a **label-stratified image-level split**, not a verified patient-level split. Although the counts match those reported in the paper, the exact sample identifiers and random seed used by the authors were unavailable.

See [`DATASET.md`](DATASET.md) for dataset preparation details.

## 6. Reproduction results

The following values were obtained by this independent implementation and are not the official results reported by the original paper.

| Model | Accuracy | Macro F1 | Weighted F1 | QWK |
|---|---:|---:|---:|---:|
| Hybrid CNN-ViT ep20 | 0.7872 | 0.6284 | 0.7886 | 0.8665 |
| Denoised Hybrid, alpha=0.25 ep20 | 0.7776 | 0.6296 | 0.7831 | 0.8747 |
| **Denoised Hybrid, alpha=0.5 ep20** | **0.8022** | **0.6423** | **0.8021** | **0.8955** |

## 7. Installation

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the local CUDA environment.

## 8. Local dataset configuration

### Windows PowerShell

```powershell
$env:APTOS_TRAIN_CSV = "D:\your-local-path\train.csv"
$env:APTOS_VAL_CSV = "D:\your-local-path\val.csv"
$env:APTOS_TEST_CSV = "D:\your-local-path\test.csv"
```

If these variables are not configured, the default relative paths are:

```text
data/aptos2019_processed/train.csv
data/aptos2019_processed/val.csv
data/aptos2019_processed/test.csv
```

## 9. Main workflow

```bash
python train_baseline.py
python train_hybrid_cnn_vit_ep20.py
python train_denoiser.py
python train_denoised_hybrid_cnn_vit_alpha05_ep20_timed.py
python test_final_ep20_models.py
python select_gradcam_cases.py
python run_gradcam_final_ep20.py
```

Checkpoints, prediction files, medical images, and generated outputs are not distributed in this repository.

## 10. Optional VST/Retinex preprocessing

```powershell
$env:APTOS_PROCESSED_DIR = "D:\your-local-processed-data"
$env:APTOS_VST_RETINEX_DIR = "D:\your-output\vst-retinex"
$env:APTOS_VST_RETINEX_MILD_DIR = "D:\your-output\vst-retinex-mild"
```

```bash
python prepare_vst_retinex_dataset.py
python prepare_vst_retinex_mild_dataset.py
```

These preprocessing experiments are exploratory ablations and should not be interpreted as exact reproductions of the original MINet preprocessing pipeline.

## 11. Citation

Please cite the original paper when using the scientific ideas implemented in this repository:

```text
Alanazi, S., Khan, S. U., Alotaibi, F. M., & Alonazi, M. (2025).
A Novel Noise Removal and Interpretable Deep Learning Model for
Diabetic Retinopathy Detection.
International Journal of Imaging Systems and Technology, 35, e70245.
https://doi.org/10.1002/ima.70245
```

## 12. License and third-party components

A source-code license has not yet been selected. Before changing this repository to public visibility, confirm code ownership and the intended license with the supervising professor or institution.

Third-party packages, code fragments, pretrained components, and model implementations remain subject to their respective licenses. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
