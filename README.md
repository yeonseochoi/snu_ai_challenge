# Qwen3-VL Video Frame Ordering

Qwen3-VL-8B-Instruct를 LoRA로 미세조정해, 문장과 섞인 비디오 프레임 4장을
시간순으로 정렬하는 코드입니다. 학습, TTA=2 추론, 산출물 검증을 하나의 CLI
흐름으로 정리했습니다.

최종 제출에 사용된 LoRA는 `checkpoint-2265`이며, 해당 제출 파일의 비공개
테스트 리더보드 점수는 **0.89528**입니다.

## 재현 범위와 기준 산출물

비공개 테스트에는 정답 라벨이 없으므로 이 저장소만으로 0.89528을 직접
계산할 수는 없습니다. 운영진 평가 서버에 추론 CSV를 제출해야 점수를 확인할
수 있습니다. 대신 최종 제출에 실제 사용된 아래 세 파일의 SHA-256을
`checksums.sha256`에 기록했습니다.

| 파일 | SHA-256 앞 12자리 |
|---|---|
| `adapter_model.safetensors` | `e00d5e137ed2` |
| `adapter_config.json` | `622be38bc0af` |
| `submission_tta2.csv` | `6b9d4bb2a90b` |

주의: 과거 `inference_test_tta2.py`는 TTA 시드에 Python `hash()`를 사용해
프로세스별로 순열이 달라질 수 있었습니다. 최종 제출 CSV 자체는 위 체크섬으로
고정해 보존합니다. 통합 `inference.py`는 `crc32(Id) + seed`를 사용하도록
수정해 이후 실행을 결정적으로 만들었습니다. 따라서 통합 코드의 새 결과와
과거 CSV가 일부 다를 수 있으며, 둘을 같은 결과라고 주장하지 않습니다.

## 저장소 구조

```text
.
├── common.py                 # 데이터/순열/프롬프트 공통 로직
├── config.yaml               # 학습 및 추론 설정
├── train.py                  # JSONL 생성 + ms-swift 학습
├── inference.py              # 결정적 TTA=2 추론
├── download_weights.py       # 공개 URL에서 가중치 다운로드 및 검증
├── verify_artifacts.py       # 최종 가중치/제출본 체크섬 검증
├── final_code.ipynb          # 당시 사용한 원본 Colab 노트북
├── requirements.txt
└── adapter_config.json       # 공개 배포용 PEFT 설정
```

## 권장 환경

- Ubuntu 22.04
- Python 3.10
- CUDA 지원 NVIDIA GPU
- 학습: A100 80 GB급 권장
- 공식 검증 서버: NVIDIA RTX 3090 24 GB 1장, CUDA 12.4,
  NVIDIA driver 550.54.15
- 추론은 24 GB VRAM에 맞춰 후보를 8개씩 나누어 계산
- 기준 학습 패키지: `ms-swift==4.4.2`,
  `transformers==5.12.1`, `datasets==4.8.4`

GPU/드라이버와 맞는 PyTorch를 먼저 설치한 뒤 나머지를 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# 환경에 맞는 CUDA PyTorch 설치: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip uninstall -y torchao
```

운영진 검증 환경은 인터넷이 차단됩니다. 설치 패키지, Qwen 기본 모델 가중치,
LoRA 가중치를 인터넷이 가능한 환경에서 미리 내려받아 검증 서버로 옮겨야
합니다. 학습과 추론 중에는 외부 API를 호출하지 않습니다.

## 데이터 준비

대회 데이터는 라이선스 때문에 저장소에 포함하지 않습니다. 운영진이 제공한
압축을 풀어 다음 구조로 배치합니다.

```text
data/
├── train.csv
├── test.csv
├── train/<Id>/<image files>
└── test/<Id>/<image files>
```

CSV에는 `Id`, `Sentence`, `Input_1`~`Input_4` 열이 필요하며, `train.csv`에는
추가로 `Answer`가 필요합니다.

## 가중치 준비

`adapter_model.safetensors`는 약 840 MB라 일반 GitHub 파일 제한(100 MB)을
초과합니다. 공개 다운로드 URL을 만든 뒤 다음처럼 받습니다. 게시 전에 아래
`<PUBLIC_WEIGHT_URL>`을 실제 GitHub Release/Hugging Face URL로 교체해야 합니다.

```bash
python download_weights.py \
  --url "<PUBLIC_WEIGHT_URL>" \
  --output checkpoints/final/adapter_model.safetensors
cp adapter_config.json checkpoints/final/adapter_config.json
```

다운로더는 파일을 최종 위치로 옮기기 전에 SHA-256을 검사합니다.

## 학습

전체 학습:

```bash
python train.py --data-dir data --output-dir outputs
```

인터넷이 차단된 환경에서는 기본 모델의 로컬 경로를 지정합니다.

```bash
python train.py \
  --data-dir data \
  --output-dir outputs \
  --model models/Qwen3-VL-8B-Instruct
```

데이터 변환과 명령 구성을 빠르게 확인:

```bash
python train.py --data-dir data --output-dir outputs \
  --smoke-test --prepare-only
```

고정 설정은 `config.yaml`에 있습니다. 원본과 동일하게 seed 42로 train의
5%를 validation으로 분리하고, 샘플당 identity 포함 4개 입력 순열로
증강하며, rank 64/alpha 128 LoRA를 1 epoch 학습합니다. `ms-swift`가 생성한
마지막 `checkpoint-*`의 adapter 파일을 `checkpoints/final/`로 복사합니다.

학습은 CUDA/cuDNN 및 병렬 커널의 영향으로 가중치 바이트 단위까지 완전히
같아진다고 보장할 수 없습니다. 공개된 최종 가중치는 결과 재현의 기준입니다.

## 테스트 추론

```bash
python inference.py \
  --data-dir data \
  --adapter-dir checkpoints/final \
  --model models/Qwen3-VL-8B-Instruct \
  --output outputs/submission_tta2.csv \
  --resume
```

`--resume`은 `outputs/submission_tta2_partial.csv`가 있을 때 완료된 ID를
건너뜁니다. 기존 24 GB GPU 전체 실행 기록은 약 12분으로, 대회 제한인
24시간 이내입니다.

## 대회 규칙 준수

- Python으로 학습 및 추론하며 추론 중 인터넷이나 외부 상용 API를 사용하지
  않습니다.
- 학습에는 운영진이 제공한 train 데이터만 사용합니다.
- 단일 `Qwen/Qwen3-VL-8B-Instruct` 모델과 하나의 LoRA adapter만 사용하며
  모델 앙상블은 사용하지 않습니다.
- LoRA와 동일 모델에 대한 TTA=2만 사용합니다.
- 테스트 데이터 또는 수작업 테스트 라벨을 학습에 사용하지 않습니다.
- 모든 입출력 경로는 CLI로 전달하는 상대경로이며 UTF-8 소스코드입니다.
- 기본 모델과 LoRA를 포함해 전체 모델 크기는 80 GB 미만이어야 합니다.

## 최종 산출물 확인

GitHub에 포함된 설정과 기준 제출본을 검사하려면:

```bash
python verify_artifacts.py \
  --adapter-dir checkpoint-2265 \
  --submission checkpoint-2265/submission_tta2.csv
```

다운로드한 가중치까지 필수로 검사하려면:

```bash
python verify_artifacts.py \
  --adapter-dir checkpoints/final \
  --submission checkpoint-2265/submission_tta2.csv \
  --require-weights
```

새 추론 결과가 과거 제출본과 바이트 단위로 같은지도 별도로 확인할 수 있습니다.

```bash
sha256sum outputs/submission_tta2.csv
# 기준: 6b9d4bb2a90bcc6a162bf58d47888e1c202315607a5179c9b653a03459a08ab8
```

## 공개 전 체크리스트

1. 가중치를 GitHub Release 또는 Hugging Face에 공개한다.
2. README의 `<PUBLIC_WEIGHT_URL>`을 실제 URL로 바꾼다.
3. 새 환경에서 다운로드, 체크섬 검증, 추론을 한 번 실행한다.
4. 저장소에 대회 원본 데이터가 포함되지 않았는지 확인한다.
5. 운영진 평가 서버에서 생성 CSV의 점수를 확인한다.
