# 결함 주입 기록 (4단계)

`designs/inject.py`로 기본 설계서 두 개에 오류 사례 18개를 무작위 배치해 맹검 변형 20개를 만들었다.
어느 파일에 무슨 결함이 있는지는 `sealed/answer_key.enc`에만 있다. 7단계 채점 전에는 열지 않는다.

- 날짜: 2026-10-03
- 보류 해제: `sealed/holdout.enc`를 메모리에서 복호화. 공개 CSV + 보류 행으로 다시 계산한 목록 해시가
  `docs/holdout_log.md`의 값과 일치 (1단계 이후 목록 변경 없음)
- **보류 6개 중 3개 조정**: 봉인 해제 뒤 설계서 형식·합성 데이터로 그대로 표현할 수 없어 사용자 승인을 받아 조정했다.
  조정 내용은 정답표 봉인본에만 있다. 7단계에서 H2를 보류 6개 전체와 조정 없는 3개로 따로 보고한다.
- 난수 시드: 정답표 봉인본에만 기록 (시드 + 공개 패치로 공개 사례 배치를 다시 계산할 수 없게)
- 정답표 암호: 보류 목록과 같은 암호 (사용자 결정)

## 배치 규칙

| 항목 | 규칙 |
|---|---|
| 변형 수 | 유형별 10개 = 결함 8 + 깨끗한 것 2 (그중 1개는 "수상해 보이지만 정당한" 변경, 정답표에 결함 아님으로 기록) |
| 변형당 결함 | 1~2개. 한 변형 안의 두 결함은 서로 다른 항목을 건드린다 |
| 유형 제한 | 각 사례의 `design_types`(고정/동적/둘다)에 맞는 유형에만 |
| 고른 배치 | 사례별 배치 횟수 차이 ≤ 1. 추가 배치는 공개·보류 비율(12:6)대로 나눔 |
| 단독 배치 | 보류 사례 1개는 다른 결함과 같은 변형에 넣지 않음 (사용자 결정) |
| 맹검 | 파일명·`design_id` = 무작위 4자리 코드. 사례 id 없음. 심은 항목 이름은 기본 설계와 같은 방식. 20개 모두 특징·포함 기준 순서를 섞음 |

## 결과

- 18개 사례 모두 1회 이상 배치. 배치 횟수: 사례당 1회 또는 2회
- **배치 합계: 공개 16회, 보류 8회** (총 24회)
- 결함 변형 16개 중 결함 1개 8개, 2개 8개
- 깨끗한 변형 4개(정당한 변경 포함 2개)는 동결된 점검기로 5,000명 데이터까지 통과 확인.
  결함 변형에는 점검기를 돌리지 않았다 (성공 기준 고정 전에 탐지 결과를 보지 않기 위해, 사용자 결정)

## 기본 설계서 (`designs/base/`)가 3단계 fixture와 다른 점

| 파일 | 3단계 fixture (`tests/fixtures/`) | 4단계 기본 설계 |
|---|---|---|
| 두 파일 공통 | `split_unit: "patient"`, `split.key`·`model.tuning.cv_key`: `patient_id` | `split_unit: "family"`, `split.key`·`model.tuning.cv_key`: `family_id` (4단계 확정 사항) |
| 두 파일 공통 | `design_id`: `clean_*` | `design_id`: `base_dynamic_aki`, `base_fixed_readmission` |
| `fixed_readmission.json` | `outcome.ascertainment` 없음 | `outcome.ascertainment: {"by_stratum": ["discharge_status"], ...}` (퇴원처별 결과 확인을 층별로 보고) |

두 기본 설계 모두 설계서 단계·데이터 단계(5,000명)에서 판정 문제 없음.
3단계 fixture와 그 시험은 바꾸지 않았다.

## 추가 데이터

- `data/synth/outpatient_visits.csv.gz` (+ `OUTPATIENT_MANIFEST.json`): 외래 방문 기록. `python -m synth.outpatient`로
  저장된 입원·환자 테이블에서 만든다 (seed 20261003). 기존 6개 테이블과 `MANIFEST.json`은 바뀌지 않았다.
- `synth.generate.load()`는 이 파일을 읽지 않는다 (기존 6개 테이블만). 외래 파일을 둔 뒤에도 두 기본 설계가
  `run_check.py --data data/synth`로 통과함을 확인했다.

## 파일 해시 (SHA-256)

| 파일 | SHA-256 |
|---|---|
| designs/base/dynamic_aki.json | `66b94dbec01fa65a1276e7453288d5a6484f42399a844229e319f4d2cf8c5743` |
| designs/base/fixed_readmission.json | `f421bf5838825961d1187c14eb120032a601b7382424112cf0f0cfd278a99e1b` |
| designs/variants/design_0B14.json | `3f530af5273682c39d94880997c42ba0a528d0147e62b7fc73124af44a9e70cc` |
| designs/variants/design_1D2D.json | `4b41bfdd0380389cb87bbe441f8492872e96b920d9dcdc44c9d7792728b03de0` |
| designs/variants/design_2340.json | `98290e4defa6e7390722b99c55bd0665751f28bdcd6dfddd6339c3766a9198b7` |
| designs/variants/design_48A5.json | `cc17412135a4c00a431755df895cff4e3f300186d6212aa1f12edfa86a32aa60` |
| designs/variants/design_4BE0.json | `6bc350c416cbd57209133c32bdd20439b8654231db8642ac3d1fabde3cccec17` |
| designs/variants/design_5350.json | `798432f875e7be8263643e6f472a8df5e227f1820a1ec567baf5ecbdfdedb2be` |
| designs/variants/design_7F4F.json | `bdea66d9a6d0e83d3d669a44c8bfee45388d4744b4656048e3913b0fb56b0bae` |
| designs/variants/design_800E.json | `52e218d49a4ad516f7e4d3c4694bc9c0111afe22d05bde93286aa0c6404fa846` |
| designs/variants/design_94F5.json | `bde11a8ab7871b5152f636f719cb5f5de1a32087c8fa649ff00f4295e239153b` |
| designs/variants/design_9672.json | `15f7eff6c08cc22d3cae080ae88a0941c68ecffc56cc5cb58f29478d2c1e9c6f` |
| designs/variants/design_9AF0.json | `c7bd9ceb4374bd88790ddae2d231585f4d1a3d68ccec59063e72cf00b756046c` |
| designs/variants/design_A1C2.json | `1826af3a9cc8880765ed33c3c15e0cc6cf158e9d857ffa27f4b7cae1c679b227` |
| designs/variants/design_AF15.json | `5e15f475258e2160d70145806e8628aadd4eb25ff4350d3c259469320bdb6b38` |
| designs/variants/design_C963.json | `3eb864d29eb6836e2d4f2b1eaac4ab0e9bde3479fe41d927449f19c5df419b58` |
| designs/variants/design_D568.json | `c2078fbca4194685c9b1f09c48313e51280f9330dbf6b3d24c4298ed267385ed` |
| designs/variants/design_D97B.json | `0baeb43094ffc4c544c6dfb0b474af8d8dd36d41c4a14b67693d318b1c78e1ba` |
| designs/variants/design_E0E0.json | `9451e850e933b80a72b65e070a46850cbccd91a38ab1d9013e48d5bdf548c89e` |
| designs/variants/design_E6D6.json | `2ce5cbd1588972f811658038961f342209c6821a2f85a284610feed37cbd3155` |
| designs/variants/design_EC59.json | `e94b4e819f8ccc215f3721ff12430d30d042edabc0a853dfa91e44203313440b` |
| designs/variants/design_F872.json | `55b9dae7b2498fe891418e5946554a8d72398008570df28233fbe856cdb1de2d` |

정답표 봉인본에도 각 변형의 해시가 들어 있어, 7단계에 열 때 변형이 바뀌지 않았는지 대조할 수 있다.
