# 단계별 계획

각 단계는 **새 세션**에서 "docs/phases.md의 N단계를 진행해"로 시작한다. 계획을 먼저 보여 주고 사용자 확인 후 실행한다. 끝나면 완료 기준을 확인하고 체크한 뒤 커밋한다.

---

## 1단계. 오류 목록과 층화 보류 추첨

**할 일**
- [ ] `tools/seal.py`: 암호(사용자 입력)로 파일을 암호화/복호화하는 도구. `cryptography` 패키지의 Fernet + PBKDF2. 평문은 커밋하지 않는다.
- [ ] 오류 사례 18개 작성 (`docs/research_plan.md` 7.1). 각 행: `id, name, source(Kapoor L?/PROBAST 영역), question(Q?), design_types(고정/동적/둘다), how_to_inject, expected_verdict(차단/경고)`.
- [ ] 질문별 층화 보류 추첨 (질문마다 1개). 시드와 날짜를 `docs/holdout_log.md`에 기록.
- [ ] 공개 12개 → `designs/error_catalog_public.csv` (커밋)
- [ ] 보류 6개 → 사용자가 정한 암호로 `sealed/holdout.enc` (커밋). 평문은 즉시 삭제.

**완료 기준**
- 공개 CSV 12행, 질문별 배분이 계획과 일치
- `sealed/holdout.enc`만 커밋되고 평문은 git 이력 어디에도 없음 (`git log -p`로 확인)
- 사용자가 암호를 안전한 곳에 적어 둠

**주의**: 이 세션은 보류 사례를 보게 되지만, 이후 세션은 기억이 없으므로 문제없다. 이 세션에서 스킬 코드를 쓰지 않는다.

---

## 2단계. 합성 EHR 생성기

**할 일**
- [ ] `synth/generate.py`: 환자 약 500명, MIMIC-IV 유사 구조
  - `patients` (patient_id, family_id 일부, 나이, 성별)
  - `admissions` (admission_id, admit_time, discharge_time, discharge_status, unit: ICU/병동)
  - `labs` (creatinine 등, collect_time, report_time — **보고 시각이 채취 시각보다 30분~수 시간 뒤**)
  - `diagnoses` (ICD 코드, **시각 없음**)
  - `procedures` (날짜만 있음)
  - `orders` (투석 오더, 신장내과 협진 등 — Q4 대리 변수용)
- [ ] 일부 환자는 크레아티닌이 상승해 KDIGO AKI 발생, 일부는 30일 내 재입원
- [ ] 퇴원 직전 채취·퇴원 후 보고되는 검사 일부 포함
- [ ] ICU는 측정이 잦고 병동은 드물게 (Q7용)
- [ ] 시드 고정, 같은 시드면 같은 데이터

**완료 기준**: 생성 → 기본 통계(AKI 비율, 재입원 비율, 환자당 입원 수) 출력, 시험 통과

---

## 3단계. 꼬리표 모듈, 점검 함수, 결정 잠금, 스킬

**할 일**
- [ ] `leakcheck/rules.py`: 규칙표 3층 (`docs/research_plan.md` 4절). 진단 코드 = 퇴원 시각, 검사 = 보고 시각, 날짜만 = 그날 23:59
- [ ] `leakcheck/tagging.py`: 모든 행에 entity, split_unit, available_time, provenance 부착. 규칙 없는 행·모순 행은 멈추고 보고
- [ ] `leakcheck/features.py`: 출처를 기록하는 특징 생성 함수 (꼬리표 상속)
- [ ] `designs/schema.json`: 설계서 형식 (tₚ 정의, 결과 정의·결과 창, 포함·제외 기준, 특징 목록, 분할 키, 전처리 단계와 적합 범위, 대리 변수 목록)
- [ ] `leakcheck/checks.py`: Q1·Q2·Q3·Q5 차단, Q4 차단/경고, Q6 기록, Q7 경고. 출력은 구조화된 JSON (`{question, verdict, target, reason}`)
- [ ] `leakcheck/lock.py`: 결정 카드(기술 통계만) + 잠금 전 성능 계산 금지
- [ ] `skill_src/leakage-check/SKILL.md` + 호출 스크립트: 에이전트가 설계서를 받으면 점검을 먼저 돌리고 결과를 보고하도록 하는 지침
- [ ] 단위 시험: **공개 12개 사례** 각각에 대해 기대 판정이 나오는지 + 깨끗한 설계는 통과하는지
- [ ] 핵심 시연 시험: 같은 진단 코드 규칙이 동적 예측에서 차단, 고정 시점 예측에서 통과

**완료 기준**: 모든 시험 통과 → 사용자 확인 → `git tag skill-frozen` → 이후 `leakcheck/`, `skill_src/` 수정 금지

**주의**: 점검은 사례별 금지 목록이 아니라 **원리(시각 비교, 분할 키 비교, 적합 범위 비교)**로 구현한다. 보류 사례를 잡을 수 있는지가 여기에 달려 있다.

---

## 4단계. 보류 해제, 결함 주입, 맹검 변형

**전제**: `skill-frozen` 태그가 있어야 시작한다.

**할 일**
- [ ] 사용자에게 암호를 받아 `sealed/holdout.enc` 복호화 (메모리에서만 사용)
- [ ] `designs/base/`: 두 유형(고정 시점 재입원, 동적 AKI)의 깨끗한 기본 설계서
- [ ] `designs/inject.py`: 18개 사례를 무작위로 배치해 유형별 변형 10개(결함 8 + 깨끗한 것 2) 생성. 변형당 결함 1~2개. 파일명은 무작위 코드
- [ ] 정답표(파일 → 심은 결함 id) → 사용자 암호로 `sealed/answer_key.enc`. 평문 삭제
- [ ] 변형 파일만 `designs/variants/`에 커밋

**완료 기준**: 변형 20개 커밋, 정답표는 암호화본만 존재, 18개 사례가 모두 최소 1회 이상 배치됨

---

## 5단계. 채점 규칙과 성공 기준 고정

**할 일**
- [ ] `experiment/scoring_rules.md`: `docs/research_plan.md` 7.5를 구체화 (탐지/미탐지/오경보 판정 예시 포함)
- [ ] `experiment/grader.py`: 에이전트 보고서 → (지목 대상, 문제 종류) 추출 → 정답표와 대조. 조건(가/나) 표시를 가린 채 채점
- [ ] `docs/success_criteria.md`: H1, H2 기준 (7.6), 날짜 기록
- [ ] 커밋 후 `git tag criteria-locked`

**완료 기준**: 가짜 보고서 몇 개로 채점기가 의도대로 판정하는지 시험

---

## 6단계. (가)/(나) 실행

**할 일**
- [ ] `experiment/run.py`: 변형마다 격리된 실행 폴더 생성. (가)만 `skill_src/leakage-check`를 스킬 위치에 복사. 같은 지시문, 같은 모델로 에이전트 실행, 보고서 저장
- [ ] 먼저 소규모 시험 실행 (변형 2개 × 조건 2 × 1회)
- [ ] 본 실행 120회 (중단 시 이어서 실행 가능하게)

**미확인 사항**: 웹 Claude Code 작업 공간 안에서 에이전트(Claude)를 반복 호출할 수 있는지 소규모 시험에서 먼저 확인한다. 안 되면 사용자의 API 키를 환경 변수로 받아 Anthropic API로 실행하는 방식으로 바꾼다.

**완료 기준**: 120개 보고서 저장, 실패한 실행 0개 또는 재실행 완료

---

## 7단계. 채점과 분석

**할 일**
- [ ] 사용자 암호로 `sealed/answer_key.enc` 복호화
- [ ] 채점 → 탐지율(공개/보류, 질문별), 오경보율, 반복 간 안정성
- [ ] H1, H2 판정 (5단계 기준 그대로)
- [ ] `results/report.md`: 표, 핵심 시연 예시, 놓친 사례 분석, 한계(합성 데이터, 부분 맹검, 같은 모델 계열)

**완료 기준**: 보고서 완성, 면접에서 쓸 한 문단 요약 포함
