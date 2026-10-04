# 이전 후 검증 기록

검증일: 2026-10-04. 로컬 Windows / Python 3.12 / PyYAML 6.0.2 환경.

| 검증 | 결과 |
|---|---|
| 원본 보존 | 7개 파일을 원본 checkout과 바이트 단위 비교, 모두 일치. SHA-256은 `source_manifest.json` |
| 회귀 테스트 | `python -m unittest discover -s tests -v` — 8개 통과 |
| Python 문법 | `python -m compileall -q scripts tests archive` — 통과 |
| 예제 전처리 | 직접 작성한 12쌍 → 학습 10 / 검증 2; 양 방향 각각 5 / 1 |
| 8B·14B 설정 및 데이터 검사 | 두 YAML 모두 `--sample-ratio 1 --check-only` 통과 |
| 추론 입력 검사 | 예제 2건 `--check-only` 통과; 모델 미로딩 |
| CLI 도움말 | 전처리·학습·추론 `--help` 실행 확인 |
| 의존성 해석 | `pip install --dry-run --ignore-installed -r requirements.txt` 성공(55개 패키지); 실제 ML 패키지 설치·실행은 미수행 |
| Git 제외 | 생성 데이터, outputs, .env, 모델 가중치 제외 확인 |

회귀 테스트는 instruction 안의 원문 복구, 빈 원문·정답 거부, 방향별 분리, 중복 원문 누출 방지, 잘못된 messages 거부, 샘플링 후 0건 방지, 학습·검증 중복 탐지, 모델별 batch 설정, 평가 ID 중복 검사를 포함합니다.

정리한 파일은 `git diff --check`를 통과했습니다. `archive/` 원본에 있던 줄 끝 공백은 출처 보존을 위해 수정하지 않았습니다. Manifest의 SHA-256은 운영체제별 줄바꿈 변환 전 Git blob 내용 기준입니다.

**검증하지 않은 범위:** ML 패키지 전체의 실제 import/학습, CUDA와 bitsandbytes 작동, 8B/14B 모델 다운로드·학습·추론, 번역 품질과 성능 수치. CPU 검사 통과를 GPU 실행이나 과거 실험 재현 성공으로 해석하지 않습니다.
