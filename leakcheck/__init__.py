"""설계 단계 누수 점검 (3단계).

- rules     규칙표 3층 (언제 알려지는가, 누구의 것인가)
- timeline  tₚ 기준 기호 시각 (설계서만으로 시각 비교)
- tagging   모든 행에 entity·split_unit·available_time·provenance 부착
- design    설계서 읽기·형식 검사
- features  출처를 기록하는 특징 생성 (꼬리표 상속)
- outcomes  결과 정의 (결정 카드의 결과율용)
- checks    Q1~Q7 점검
- lock      결정 카드와 결정 잠금

판정은 모두 규칙 기반 코드가 한다. 사례별 금지 목록은 두지 않는다.
"""
