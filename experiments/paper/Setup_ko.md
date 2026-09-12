# IV-A Setup and Protocol 초안 해설 (한글)

`setup.tex`는 멘토님이 맡은 절의 초안입니다. 멘토님이 채워야 할 것은 대괄호
[ ]와 `%PENDING` 주석으로 표시했고, 나머지 숫자는 전부 2026-09-11 export의
`summary.json` 인자와 CSV, 그리고 멘토님의 attention_backend.md에서 왔습니다.
분량은 본문 약 600단어와 백본 표 하나로, 계획상 IV-A 몫인 한 쪽에 맞습니다.

## 문단별

**Backbones.** 백본 일곱 개(SimplerEnv 여섯, LIBERO 셋, OpenVLA와 UniVLA는
둘 다)와 각 벤치마크용 공개 체크포인트. 파라미터 수는 원 논문 보고값.
"Decoder depth"는 depth pruning이 점수를 매기는 층 묶음이고, CronusVLA만
언어 모델이 아니라 12층 DiT 행동 디코더임. "Actions per call"은 한 번 호출이
몇 스텝을 담당하는지로, UniVLA만 5(SimplerEnv)와 10(LIBERO)이고 나머지는 1.
attention 구현은 멘토님 문서 그대로(CogACT, CronusVLA, MiniVLA는 transformers
버전이 고른 SDPA, OpenVLA와 UniVLA는 sdpa 요청, SpatialVLA는 텍스트 디코더가
eager인 혼합, SmolVLA는 torch SDPA 직접 호출). GPU 열과 라이브러리 버전은
멘토님 몫.

**Environments and episodes.** WidowX 과제 4개 × 50, 상한 60(가지만 120).
Fractal 과제 5개 × 50, 상한 80. LIBERO suite 4개 × 과제 10개 × 10, 상한 520,
300, 280, 220. 모든 설정이 같은 과제 인스턴스와 초기 상태를 쓰고 seed는 42 +
에피소드 번호. 실패는 상한까지 감.

**Configurations.** dense + 설정 13개. 값은 전부 실행 인자에서 확인: keep
0.2/0.5, k 2/4, 예산 1/2/4, reuse preset 세 개의 문턱값(0.01/0.03/0.995,
0.015/0.04/0.99, 0.02/0.05/0.98, 이동 바닥값 0.01, 연속 상한 1/1/2), fusion 세
설정(keyframe 3, 비율 0.5 / 같음 + attention / keyframe 2, 비율 0.25, 강제
keyframe 0.03), 공통값(움직임 0.01, 엔트로피 보호 0.15, attention 보호 0.2,
반경 1). 합계 22쌍 × 14설정, 47,500 에피소드. Methods에는 숫자를 안 쓰기로
했으므로 숫자는 여기에만 나옴.

**Measures.** 성공률, 스텝당 벽시계 시간(에피소드 시간 ÷ 스텝), 호출당
시간(IV-C용), 평균 스텝. 표는 가족별 최고 성공률 설정(동률이면 스텝 적은
쪽)이고 행 이름은 가족 이름.

**Pairing and significance.** 검정을 쉬운 말로 한 문장. "같은 에피소드에서
실패→성공, 성공→실패로 뒤집힌 수를 세고, 그 뒤집힘에 대한 exact two-sided
McNemar test가 p < 0.05이면 significant라고 부른다." McNemar라는 이름은 논문
전체에서 여기 한 번. seed 하나, 실행 한 번이라는 점도 여기서 말함.

**Hardware and implementation.** GPU는 멘토님 몫. SmolVLA가 두 구현체로
돌았다는 사실과 지연 시간은 첫 그룹 안에서만 비교한다는 것, CogACT는
attention 통로가 없어 task-aware가 motion-entropy와 같다는 것.

## 2026-09-11 비평 반영 (여섯 항목)

1. ID/OOD: Environments 끝에 "모든 평가는 in-distribution"이라는 문장 추가.
   근거는 에피소드 파일의 `benchmark_protocol = released_bridge_visual_matching`과
   벤치마크별 체크포인트.
2. SmolVLA 스택 혼재: 멘토님 질문 6번(재실행 또는 지연 시간 삭제)으로 처리.
3. 다중 비교: Pairing 문단 끝에 한 문장(우연 통과 약 다섯 개, FDR 보정 통과
   하락 16칸, 상승 0칸). 보정 이름은 논문에서 여기 한 번.
4. 보정 프레임: "from episodes disjoint from the test episodes"로 씀.
   CronusVLA 기록(seed 10000, 같은 과제 목록)과 맞고, 나머지 백본은 Methods의
   주장 이상을 하지 않음. 백본별 출처는 멘토님 몫(%PENDING).
5. DiT에서 Block Influence: Backbones 문단에 정의가 residual block 일반에
   적용된다는 것과 CronusVLA 행이 그 전이를 시험한다는 문장 추가.
6. VRAM: 어느 파일에도 메모리 기록이 없어 지금은 못 씀. Limitations에 한 줄,
   재실행 때 `torch.cuda.max_memory_allocated` 로그 부탁을 다음 답장에 넣을 것.

## 멘토님이 채울 것

1. 백본별 GPU(표의 GPU 열)와 transformers, torch 버전.
2. depth pruning 보정 프레임을 어디서 뽑았는지("[drawn from ...]").
3. SmolVLA 재실행 여부에 따라 Hardware 문단의 SmolVLA 문장 유지 또는 삭제.
4. 세 재실행이 오면 Configurations의 에피소드 수는 그대로(교체이므로).

## 주의

- `tablebackbones.tex`(옛 세 백본 표)는 같은 label을 쓰므로 함께 포함하면
  안 됨. setup.tex 안의 표가 대체본.
- SmolVLA 디코더 깊이는 32 (SmolLM2). SmolVLM2-500M의 텍스트 디코더는
  SmolLM2-360M이고 config의 num_hidden_layers=32 (웹 확인). depth pruning이
  자른 인덱스가 30까지 나오는 것과, 그 층을 자르니 성공률이 14～33점 떨어지는
  것(=실제 계산 경로에 있음) 모두 32와 일치. 옛 "16 (SmolVLM)"은 SmolVLA가
  앞 16층만 쓴다는 노트에서 온 오기였음. SmolVLA는 Block Influence 아닌
  고정 인덱스로 잘랐다는 점(calibrated:false)은 IV-A에 caveat로 명시됨.
