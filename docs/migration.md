# Translate-Project → Qwen3 연구 저장소 이전 기록

이전일: 2026-10-04

원본 저장소: `johangjoo/Translate-Project`

기준 커밋: `19dad8fe171de62d14da1147b0f4b51fd1ab1d92`

## 가져온 코드

| 원본 경로 | 실행용 경로 | 처리 |
|---|---|---|
| `LoRAScript/prepare_dataset_flexible.py` | `scripts/preprocess.py` | 원본 JSON 필드와 방향 분기 추출 |
| `LoRAScript/transJson.py` | `scripts/common.py`, `scripts/preprocess.py` | system prompt, 방향 태그, messages 형식 유지; 원문 전달 수정 |
| `valida.py` | `scripts/preprocess.py` | 방향별 분리 개념 유지; CLI의 비율·seed 사용 |
| `LoRAScript/train_qwen_lora_sec.py` | `scripts/train.py`, `configs/qwen3_8b_lora.yaml` | 8B 학습 본체·설정 추출 |
| `LoRAScript/train_qwen_14b_lora.py` | `scripts/train.py`, `configs/qwen3_14b_lora.yaml` | 14B 설정 분리, 중복 학습 본체 통합 |
| `LoRAScript/inference_lora_improved.py` | `scripts/inference.py` | NF4 로딩·생성 흐름 추출; Base/adapter 비교 CLI로 정리 |
| `LoRAScript/requirements.txt` | `requirements.txt` | 원본은 archive에 보존; 새 실행 대상과 구분 |

위 7개 원본 파일은 `archive/`에 내용 변경 없이 보존했습니다.
`api/translation/qwen_local.py`의 chat template 사용도 검토했습니다. 앱용 자막 프롬프트·few-shot 예시·후처리는 옮기지 않고, 학습 messages와 동일한 프롬프트로 통일했습니다.
새 CLI, 데이터 검사, 중복 제거, 예제, 테스트, 설명 문서는 이번 정리에서 추가한 부분입니다. 과거 구현·실험 결과로 표기하지 않습니다.

## 제외한 내용

- Electron, `node_modules`, 영상·음성 처리, Whisper/pyannote, FFmpeg, Cloudflare 및 FastAPI 앱 코드
- `train_qwen_lora.py`: 8B 코드와 중복되고 실행 측정으로 확인되지 않은 속도 추정 출력 포함
- `LoRAScript/config.ini`: 학습 스크립트에서 읽지 않으며 max length 2048/epoch 3 등 실제 코드와 불일치
- 진단용 대화형 스크립트 및 앱 통합 테스트: 현재 CLI와 무관한 경로·프롬프트 의존
- 원본 전체 README: 앱 설치 설명 등 주제가 달라 새 연구용 README 작성

## 수정한 동작과 의미

1. 원본 prepare 스크립트는 원문을 `instruction` 안에 넣고 `input`은 빈 문자열로 저장합니다. 원본 transJson은 `input`만 읽습니다. 두 스크립트를 그대로 연결하면 user 메시지에 원문이 빠집니다. 새 전처리는 원본 필드를 바로 변환하거나 instruction의 `Korean:`/`Japanese:` 뒤 원문을 읽습니다. 이 문제만으로 과거 실험 데이터가 실제 손상됐다고 판단할 수는 없습니다.
2. 원본 첫 분리는 순서대로 95:5이고, `valida.py`는 방향별 35,000개를 따로 뽑습니다. 새 CLI의 기본값 90:10은 재구성한 절차입니다. 과거 실제 분리 비율로 주장하지 않습니다.
3. 같은 방향의 동일 원문을 한 번만 사용하고 방향별로 shuffle/split합니다. 기존 데이터 파일을 덮어쓰지 않습니다. 파싱 오류는 중단하며, 형식 오류 레코드는 통계에 기록하고 제외합니다.
4. 원본 8B/14B batch 설정 차이(6×3 vs 1×16), 샘플 비율 0.1, LoRA 및 SFT 설정은 YAML에 보존했습니다.
5. 모델 자동 로딩을 main 안으로 옮기고 CPU 검증 모드를 추가했습니다. 새 학습 경로는 단일 CUDA GPU를 대상으로 합니다.
6. 양자화된 모델에 LoRA를 자동 병합하는 대신 어댑터를 저장합니다. 추론은 동일한 Base 위에 어댑터를 장착합니다.
7. 과거 추론의 수동 ChatML 및 화자 프롬프트를 학습 형식과 같은 chat template로 맞췄습니다. 출력 토큰만 decode하고 Base와 LoRA의 생성 조건을 고정합니다. 새 추론 결과는 과거 앱 출력과 동일하다고 보장하지 않습니다.

## 아직 확보되지 않은 자료

- 정확한 원본 데이터셋 식별자·버전·사용 조건, 실제 정제 전후 건수
- 실제 train/validation 분리 파일, 추가 외부 데이터 출처
- 최종 학습 환경의 freeze 파일, GPU와 실행 시간, checkpoint 및 trainer 로그
- 보고서 원본과 8B/14B Base·LoRA 비교 화면, 평가 입력·출력

저장소에 없는 사실은 이전 대화의 설명만으로 확정하지 않았습니다. 특히 8B/14B의 개선 효과를 수치·예시로 만들어 넣지 않았습니다.
