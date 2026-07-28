# Qwen3-VL Video Frame Ordering

Qwen3-VL-8B-Instruct를 LoRA로 미세조정하여, 주어진 문장과 무작위로 섞인 비디오 프레임 4장을 시간순으로 정렬하는 프로젝트입니다.

학습 데이터 생성, LoRA 학습, TTA=2 추론, 제출 파일 생성 및 최종 산출물 검증 과정을 CLI로 실행할 수 있습니다.

최종 제출에는 `checkpoint-2265`의 LoRA 가중치를 사용했으며, 비공개 테스트 리더보드 점수는 **0.89528**입니다.

## 저장소 구조

```text
.
├── common.py
├── config.yaml
├── train.py
├── inference.py
├── download_weights.py
├── verify_artifacts.py
├── requirements.txt
├── adapter_config.json
├── checksums.sha256
└── checkpoint-2265/
    ├── adapter_config.json
    └── submission_tta2.csv
```

각 파일의 역할은 다음과 같습니다.

- `common.py`: 데이터 처리, 입력 순열 및 프롬프트 공통 로직
- `config.yaml`: 학습 및 추론 설정
- `train.py`: 학습 데이터 생성 및 LoRA 학습
- `inference.py`: TTA=2 추론 및 제출 파일 생성
- `download_weights.py`: LoRA 가중치 다운로드 및 SHA-256 검증
- `verify_artifacts.py`: 가중치, 설정 및 기준 제출 파일 검증
- `requirements.txt`: Python 라이브러리 목록
- `adapter_config.json`: 최종 LoRA 설정
- `checksums.sha256`: 최종 산출물의 SHA-256
- `checkpoint-2265/submission_tta2.csv`: 점수 0.89528을 기록한 기준 제출 파일

## 실행 환경

최종 모델은 다음 Google Colab 환경에서 학습했습니다.

- Google Colab Linux
- Python 3.12
- NVIDIA A100-SXM4 40GB
- `torch==2.11.0+cu128`
- CUDA 12.8 PyTorch 빌드
- `ms-swift==4.4.2`
- `transformers==5.12.1`
- `datasets==4.8.4`
- `peft==0.19.1`

기존 전체 테스트 추론은 NVIDIA RTX 4090 24GB 환경에서 실행했습니다. 운영진 환경에서는 Python 3.12와 CUDA 지원 NVIDIA GPU를 사용하고, GPU 및 CUDA 환경에 맞는 PyTorch 2.11.0 빌드를 설치합니다.

24GB VRAM 환경에서 실행할 수 있도록 추론 시 24개 후보 순열을 8개씩 나누어 계산합니다.

## 설치

먼저 GPU 및 CUDA 환경에 맞는 PyTorch를 설치한 후 나머지 라이브러리를 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# 실행 환경에 맞는 PyTorch 설치
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
pip uninstall -y torchao
```

대회 규정에 따라 학습 및 추론 코드는 인터넷 연결 없이 실행할 수 있어야 합니다.

따라서 오프라인 환경에서 실행하기 전에 다음 항목을 준비해야 합니다.

- `requirements.txt`에 명시된 Python 패키지
- `Qwen/Qwen3-VL-8B-Instruct` 기본 모델 가중치
- GitHub Release에서 제공하는 LoRA 가중치
- 운영진이 제공한 대회 데이터

필요한 파일을 준비한 이후의 학습 및 추론 과정에서는 인터넷이나 외부 API를 사용하지 않습니다.

## 데이터 준비

대회 데이터는 저장소에 포함하지 않습니다.

운영진이 제공한 데이터를 다음 구조로 배치합니다.

```text
data/
├── train.csv
├── test.csv
├── train/
│   └── <Id>/
│       └── <image files>
└── test/
    └── <Id>/
        └── <image files>
```

CSV에 필요한 열은 다음과 같습니다.

- 공통: `Id`, `Sentence`, `Input_1`, `Input_2`, `Input_3`, `Input_4`
- 학습 데이터 추가 열: `Answer`

## 기본 모델 준비

사용한 기본 모델은 다음과 같습니다.

```text
Qwen/Qwen3-VL-8B-Instruct
```

공식 모델 페이지:

https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

인터넷이 가능한 환경에서 기본 모델을 미리 다운로드한 후 다음과 같이 배치합니다.

```text
models/
└── Qwen3-VL-8B-Instruct/
```

학습 및 추론 명령의 `--model` 인자에는 위 로컬 경로를 전달합니다.

## LoRA 가중치 준비

최종 제출에 사용한 LoRA 가중치는 GitHub Release에서 직접 다운로드할 수 있습니다.

https://github.com/yeonseochoi/snu_ai_challenge/releases/download/v1.0.0/adapter_model.safetensors

다음 명령을 실행하면 가중치를 다운로드하고 SHA-256을 자동으로 확인합니다.

```bash
python download_weights.py \
  --url "https://github.com/yeonseochoi/snu_ai_challenge/releases/download/v1.0.0/adapter_model.safetensors" \
  --output checkpoints/final/adapter_model.safetensors

cp checkpoint-2265/adapter_config.json \
  checkpoints/final/adapter_config.json
```

다운로드한 파일은 체크섬 검증을 통과한 경우에만 최종 경로에 저장됩니다.

최종 LoRA 가중치의 SHA-256은 다음과 같습니다.

```text
e00d5e137ed29a8487c963c534b52ce9196faa489517fb00e24fd2f3a5554513
```

## 학습

전체 학습은 다음 명령으로 실행합니다.

```bash
python train.py \
  --data-dir data \
  --output-dir outputs \
  --model models/Qwen3-VL-8B-Instruct
```

전체 학습 전에 데이터 처리 과정만 간단히 확인하려면 다음 명령을 실행합니다.

```bash
python train.py \
  --data-dir data \
  --output-dir outputs \
  --model models/Qwen3-VL-8B-Instruct \
  --smoke-test \
  --prepare-only
```

주요 학습 설정은 `config.yaml`에 저장되어 있습니다.

- Random seed: 42
- Validation 비율: Train 데이터의 5%
- 데이터 증강: 원본 순열을 포함한 샘플당 4개 입력 순열
- LoRA rank: 64
- LoRA alpha: 128
- Epoch: 1
- Base model: `Qwen/Qwen3-VL-8B-Instruct`

학습이 완료되면 `ms-swift`가 생성한 최종 `checkpoint-*` 디렉터리에서 다음 파일을 복사합니다.

```text
adapter_model.safetensors
adapter_config.json
```

복사 예시는 다음과 같습니다.

```bash
mkdir -p checkpoints/final

cp outputs/<checkpoint-directory>/adapter_model.safetensors \
  checkpoints/final/adapter_model.safetensors

cp outputs/<checkpoint-directory>/adapter_config.json \
  checkpoints/final/adapter_config.json
```

CUDA, cuDNN 및 GPU 연산의 비결정성으로 인해 새로 학습한 가중치가 공개 가중치와 바이트 단위로 완전히 일치하지 않을 수 있습니다.

최종 제출 결과를 확인할 때는 GitHub Release에 공개된 LoRA 가중치를 사용하는 것을 권장합니다.

## 테스트 추론

테스트 추론은 다음 명령으로 실행합니다.

```bash
python inference.py \
  --data-dir data \
  --adapter-dir checkpoints/final \
  --model models/Qwen3-VL-8B-Instruct \
  --output outputs/submission_tta2.csv \
  --resume
```

최종 제출 파일은 다음 위치에 저장됩니다.

```text
outputs/submission_tta2.csv
```

`--resume` 옵션을 사용하면 이전 실행에서 완료된 샘플을 건너뛰고 추론을 이어서 진행합니다.

중간 결과는 다음 위치에 저장됩니다.

```text
outputs/submission_tta2_partial.csv
```

기존 RTX 4090 24GB 환경에서 전체 테스트 추론에는 약 12분이 소요되었습니다.

운영진의 RTX 3090 24GB 환경에서도 실행할 수 있도록 후보 순열을 8개씩 나누어 계산하도록 구성했습니다.

## TTA 설정과 재현성

추론에는 동일한 기본 모델과 동일한 LoRA adapter를 이용한 TTA=2를 적용합니다.

과거 제출에 사용한 추론 코드에서는 TTA 순열 생성에 Python의 `hash()`를 사용했습니다. Python의 `hash()`는 실행 프로세스에 따라 값이 달라질 수 있으므로, TTA 순열 역시 실행마다 달라질 가능성이 있었습니다.

현재 `inference.py`에서는 다음 값을 이용해 ID별 TTA 순열을 결정적으로 생성합니다.

```text
crc32(Id) + seed
```

따라서 현재 통합 코드로 생성한 결과와 과거 최종 제출 CSV가 일부 다를 수 있습니다.

비공개 테스트 리더보드 점수 0.89528에 사용된 제출 파일은 다음 경로에 별도로 보존했습니다.

```text
checkpoint-2265/submission_tta2.csv
```

## 최종 산출물 검증

최종 제출에 사용된 산출물의 SHA-256은 다음과 같습니다.

| 파일 | SHA-256 |
|---|---|
| `adapter_model.safetensors` | `e00d5e137ed29a8487c963c534b52ce9196faa489517fb00e24fd2f3a5554513` |
| `adapter_config.json` | `513a8db6560852afd5e03f6bad8a5d624607916289c7da18dcda924dad8a743e` |
| `submission_tta2.csv` | `6b9d4bb2a90bcc6a162bf58d47888e1c202315607a5179c9b653a03459a08ab8` |

저장소에 포함된 설정 파일과 기준 제출 파일은 다음 명령으로 확인할 수 있습니다.

```bash
python verify_artifacts.py \
  --adapter-dir checkpoint-2265 \
  --submission checkpoint-2265/submission_tta2.csv
```

다운로드한 LoRA 가중치까지 함께 확인하려면 다음 명령을 실행합니다.

```bash
python verify_artifacts.py \
  --adapter-dir checkpoints/final \
  --submission checkpoint-2265/submission_tta2.csv \
  --require-weights
```

새로 생성한 제출 파일의 체크섬은 다음 명령으로 확인할 수 있습니다.

```bash
sha256sum outputs/submission_tta2.csv
```

비공개 테스트 데이터에는 정답이 포함되어 있지 않기 때문에 이 저장소만으로 리더보드 점수 0.89528을 직접 계산할 수는 없습니다.

점수를 확인하려면 생성한 제출 CSV를 운영진 평가 시스템에 제출해야 합니다.

## 대회 규칙 준수 사항

- Python 기반 학습 및 추론
- 운영진이 제공한 Train 데이터만 학습에 사용
- 외부 데이터 및 수작업 테스트 라벨 미사용
- 학습 및 추론 중 외부 상용 API 미사용
- 단일 `Qwen/Qwen3-VL-8B-Instruct` 모델 사용
- 단일 LoRA adapter 사용
- 모델 앙상블 미사용
- 동일 모델에 대한 TTA=2 적용
- 데이터 입출력 경로를 CLI 인자로 전달
- UTF-8 소스코드 사용
- 기본 모델과 LoRA를 포함한 전체 모델 크기 80GB 미만

## 모델 크기

추론에 필요한 모델 파일의 크기는 다음과 같으며, 합계는 80GB 제한보다 작습니다.

| 구성 | 크기 |
|---|---:|
| `Qwen/Qwen3-VL-8B-Instruct` 기본 모델 | 약 17.5GB |
| `adapter_model.safetensors` LoRA 가중치 | 840,009,216 bytes (약 0.84GB) |
| 합계 | 약 18.34GB |

LoRA 가중치 크기는 GitHub Release에 게시한 파일을 기준으로 측정했습니다. 기본 모델 크기는 공식 Hugging Face 모델 저장소의 표시값을 기준으로 합니다.
