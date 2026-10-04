# Base vs LoRA 평가

현재 실제 비교 출력·점수는 없습니다. 아래 절차와 모델별 문서는 평가를 위한 준비물입니다.

## 비교 실행

학습에 사용하지 않은 입력을 `id`, `direction`(`ko2ja` 또는 `ja2ko`), `source`, 선택적 `reference` 필드의 JSONL로 준비합니다. `sample_inputs.jsonl`은 직접 작성한 실행 예제 2개이며 벤치마크가 아닙니다.

```powershell
python scripts/inference.py --model Qwen/Qwen3-8B --adapter outputs/qwen3-8b-lora-10ratio/lora_adapters --input evaluation/sample_inputs.jsonl --output outputs/8b_comparison.jsonl
python scripts/inference.py --model Qwen/Qwen3-14B --adapter outputs/qwen3-14b-lora-10ratio/lora_adapters --input evaluation/sample_inputs.jsonl --output outputs/14b_comparison.jsonl
```

입력·시스템 프롬프트·chat template·thinking 비활성화·NF4 설정·생성 설정을 두 조건에 동일하게 적용합니다. 각 문장의 seed도 동일하게 초기화합니다. 모든 Base 출력을 생성한 뒤 adapter를 로드하여 LoRA 출력을 생성합니다. GPU/라이브러리 차이에 따른 완전한 비트 수준 재현은 보장하지 않습니다.

생성 설정은 원본 추론의 temperature 0.7 / top_p 0.9 / repetition_penalty 1.1 / 최대 512토큰을 기반으로 하고 top_k 50, seed 42를 명시했습니다. 새 비교 절차이며 과거 앱의 화면 결과를 재현했다고 주장하지 않습니다.

## 기록할 내용

- 모델·어댑터 출처, 실행 날짜, 패키지 버전, 데이터 분리와 생성 조건
- 동일 원문에 대한 Base/LoRA 실제 출력, 참조 번역과 평가 근거
- 의미 보존, 자연스러움, 존댓말·문체, 누락·추가·환각, 번역 외 설명 발생
- 개선 사례와 함께 악화·변화 없음 사례, 표본 수와 선택 방법

`8b_base_vs_lora.md`, `14b_base_vs_lora.md`에 실제 결과를 채웁니다. 수치 평가(BLEU/chrF/COMET 등)는 아직 구현·실행하지 않았습니다. 향후 고정된 별도 테스트셋과 평가 버전을 정한 후 추가합니다.

8B/14B 원본의 유효 배치 크기가 다르므로 모델 크기만의 효과로 해석하지 않습니다. 이전 대화의 개선 관찰은 보고서 원문·비교 출력 확인 후 근거와 함께 기록합니다.
