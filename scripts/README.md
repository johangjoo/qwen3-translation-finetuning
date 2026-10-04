# 실행 코드

- `preprocess.py`: 기존 전처리 3개 스크립트의 형식 변환·방향별 분리를 통합
- `train.py`: 기존 8B/14B 학습 본체를 YAML 설정으로 분리
- `inference.py`: 동일 프롬프트로 번역 LoRA 적용 전후 비교
- `common.py`: messages 형식과 입력 검증 공유

각 CLI는 `--help`를 지원합니다. 실행 예제는 루트 README를 참고하세요. 원본은 `archive/`, 변경 근거는 `docs/migration.md`에 있습니다.
