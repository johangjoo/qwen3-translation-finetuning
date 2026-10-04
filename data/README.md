# 데이터

`sample_raw.jsonl`과 `sample_train.jsonl`은 이번 저장소 정리를 위해 직접 작성한 12개 문장쌍입니다. 실제 학습 말뭉치의 일부가 아니며 번역 성능의 증거로 사용하지 않습니다.

원본 코드에서 확인한 JSON 필드:

```json
{"origin_lang":"한국어","tl_trans_lang":"일본어","tc_text":"창문을 조금 열어 주세요.","tl_trans_text":"窓を少し開けてください。"}
```

전처리 출력은 `system`, 방향 태그+원문의 `user`, 정답 번역의 `assistant`로 이루어진 messages JSONL입니다. 화자의 성별·연령 등 메타데이터는 새 실행 경로에서 사용하지 않습니다.

## 실제 데이터 준비

원본 데이터는 본인이 사용 권한을 확보한 경로로 준비하고 `data/raw/`에 로컬로 보관합니다. 현재 저장소에는 다운로드 자동화나 특정 데이터셋 링크가 없습니다. 이전 대화에 언급된 AI Hub 말뭉치의 정확한 데이터셋 ID·버전과 추가 데이터의 출처는 아직 확인해야 합니다.

```powershell
python scripts/preprocess.py --input data/raw --output-dir data/processed_full --validation-ratio 0.1 --seed 42
```

- 폴더 내 JSON(객체 또는 객체 배열)과 JSONL, 또는 단일 파일을 지원합니다. 중첩된 공급자별 JSON 구조는 먼저 확인해야 합니다.
- 기존 `instruction/input/output` JSONL도 지원합니다. 빈 input이면 instruction의 언어 라벨 뒤에서 원문을 복구합니다.
- 형식 오류/빈 문장은 제외해 수를 기록하고, 잘못된 JSON 구문은 오류로 중단합니다.
- 같은 방향의 동일 원문 중복은 첫 번째만 남깁니다. 다른 정답을 갖는 동일 원문도 이 규칙을 적용합니다.
- 방향별 seed 42 셔플 후 10%를 검증으로 분리합니다. 작은 데이터는 각 방향에서 가능하면 검증 1개 이상·학습 1개 이상을 남기므로 실제 비율이 달라질 수 있습니다.
- 원본을 덮어쓰지 않습니다. 출력 파일이 있으면 새 출력 폴더를 지정해야 합니다.
- `preprocess_stats.json`에 입력·제외·중복·분리 건수를 기록합니다.

이 분리 절차는 이번에 정리한 절차이며 과거 실험의 정확한 분리 방식을 재현했다고 주장하지 않습니다. 대규모 전처리는 데이터를 메모리에 모으므로 충분한 RAM이 필요합니다.

실제 모델 학습 전에는 YAML의 경로를 `data/processed_full/train.jsonl`, `data/processed_full/validation.jsonl`로 수정하세요. 원본 코드의 `sample_ratio: 0.1`은 **분리 후 각 파일에서 다시 10%를 학습/검증에 사용**한다는 뜻입니다.
