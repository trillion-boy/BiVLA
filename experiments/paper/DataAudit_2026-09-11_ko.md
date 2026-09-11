# 멘토님 데이터(2026-09-11 zip) 전수 점검 보고서 (한글판)

영문 원본은 `DataAudit_2026-09-11.md`이고, 이 문서는 같은 내용을 처음 보는
사람도 따라올 수 있게 풀어 쓴 것입니다. 숫자는 영문판과 같습니다.

## 0. 한 줄 요약

zip 안의 파일 1,724개를 하나도 빠짐없이 열어 확인했습니다. 파일끼리 서로
어긋나는 곳은 없었지만, "이름표와 실제 실행이 다른" 실험이 몇 군데 있어서
표에 각주를 달거나 일부를 다시 돌려야 합니다. 다시 돌려야 하는 양은 전체의
4 % 미만입니다(6절).

## 1. 무엇을 봤고, 정말 다 봤는지

**zip 구성.** 폴더 872개, 파일 1,724개. 파일은 네 종류뿐입니다.

| 종류 | 개수 | 내용 |
|---|---|---|
| `summary.json` | 699 | 한 실험 묶음(백본 x 설정 x 과제)의 요약. 성공 수, 평균 스텝, 실행 인자, 체크포인트 정보 |
| `episodes.jsonl` | 699 | 같은 묶음의 에피소드별 기록. 한 줄이 에피소드 하나(성공 여부, 스텝 수, 모델 호출 수, 시간). 전체 47,500줄 |
| `.csv` | 25 | 멘토님이 만든 표용 요약. 백본별 22개 + 환경별 `summary.csv` 3개 |
| `.lock` | 301 | 실험 스케줄러가 남긴 0바이트 빈 파일. 내용 없음 |

**셀프 검증 결과.** 마지막에 별도 스크립트로 1,724개 파일을 전부 다시 열어
다음을 확인했습니다. 결과 파일은 `experiments/data_audit_2026-09-11/file_manifest.csv`
(파일 1,724개 각각의 경로, 종류, 크기, sha256 해시, 파싱 결과)입니다.

- `summary.json` 699개 모두 JSON으로 정상 파싱.
- `episodes.jsonl` 699개 모두 한 줄씩 파싱, 합계 47,500줄, 깨진 줄 0.
- CSV 25개 모두 파싱, 합계 439행.
- `.lock` 301개 모두 0바이트(읽을 내용이 없음).
- 데이터 파일 1,423개(699 + 699 + 25)가 우리 깃허브 복사본과 바이트 단위로
  동일. 누락 0, 차이 0.

즉 "1,724개 중 내용이 있는 파일은 1,423개이고, 그 1,423개를 모두 파싱해서
분석에 썼다"가 정확한 표현입니다. 나머지 301개는 빈 파일이라 볼 내용이 없고,
깃허브에는 올리지 않았습니다.

**분석 방법.** 서로 다른 관점의 읽기 전용 점검 네 개를 각각 독립적으로
돌렸습니다(실행 설정, CSV, 에피소드 단위 구조, 세 파일 종류 간 일관성과 논문
Methods 문장 대조). 그 위에 제가 핵심 숫자(크래시 수, 스텝 상한, GPU 종류,
CronusVLA depth 행)를 파일에서 직접 다시 뽑아 대조했습니다.

## 2. 먼저 알아둘 용어

- **에피소드**: 로봇이 과제 하나를 처음부터 끝까지 시도한 한 번. WidowX는
  과제 4개 x 50회 = 200, Fractal은 5 x 50 = 250, LIBERO는 suite마다 세부
  과제 10개 x 10회 = 100.
- **스텝 상한**: 이 스텝 수 안에 성공 못 하면 실패로 끝내는 한계. WidowX
  60(가지 넣기만 120), Fractal 80, LIBERO Long 520, Goal 300, Object 280,
  Spatial 220.
- **preset**: guarded reuse의 문턱값 묶음. strict(엄격) < moderate(중간)
  < aggressive(느슨) 순으로 재사용이 잘 일어나게 되어 있음.
- **gate(게이트)**: guarded reuse가 "이번 스텝은 모델 안 돌리고 지난 행동을
  반복해도 된다"고 판단하는 조건. 게이트가 안 열리면 원래 정책과 똑같이 돈다.
- **keyframe**: temporal fusion에서 몇 번에 한 번씩 강제로 전체를 새로
  계산하는 호출. 간격 3이면 세 번에 한 번.
- **OOM**: GPU 메모리 부족(out of memory) 오류로 프로그램이 중간에 죽는 것.
- **구현체**: 같은 모델을 돌리는 코드 버전. 코드가 다르면 속도 비교가 성립하지
  않는다.
- **per-step wall-clock**: 에피소드 총 시간을 스텝 수로 나눈 값. 우리 표의
  Latency 열이 이것.

## 3. 통과한 검사 (문제 없는 부분)

- `summary.json`의 집계값(에피소드 수, 성공 수, 평균 스텝, 평균 호출 수,
  평균 시간 등)은 `episodes.jsonl`을 다시 합산한 값과 699개 모두 일치.
- CSV 307행 전부가 해당 폴더들의 합과 일치. `summary.csv` 132행은 백본별
  CSV 행과 글자 단위로 같음.
- 난수 seed = 42 + 에피소드 번호가 41,200건 모두 일치(CronusVLA는 seed를
  기록하지 않음). 모든 설정이 같은 에피소드 집합을 씀. 그래서 "같은
  에피소드끼리 짝지어 비교"가 가능함.
- 실행 상태는 699개 모두 "completed".
- depth pruning의 층 선택 규칙(앞 1/4 보호, 마지막 층 보호, 인접 층 금지,
  영향도 낮은 순)이 기록된 138개 폴더 모두에서 그대로 재현됨.
- action repeat의 호출 수 = 올림(스텝 / (chunk x k))이 2,600 에피소드 모두
  성립. foveation의 keep 비율 0.2 / 0.5도 전 폴더에 기록됨.
- guarded reuse의 연속 재사용 상한이 8,600 에피소드 모두 지켜짐.
- 크래시가 아닌 실패 21,495개는 전부 스텝 상한에서 끝남(중간에 멈춘 실패
  0). 두 시뮬레이터 모두 실패하면 끝까지 돌린다는 뜻.

## 4. 문제가 되는 발견 (중요한 순서)

각 항목에 "무슨 일인지, 왜 문제인지, 어디에 영향인지, 어떻게 할지"를
적었습니다.

**F1. UniVLA WidowX task-aware fusion, 200개 중 81개가 OOM으로 죽음.**
- 무슨 일: 가지 넣기 42개, 큐브 쌓기 39개 에피소드가 GPU 메모리 부족으로
  5~55스텝 만에 죽었고, 그대로 "실패"로 집계됨.
- 왜 문제: CSV에는 48.0 %(원래 정책 87.5 %)로 적혀 있는데, 이는 모델이
  못한 게 아니라 프로그램이 죽은 것. 안 죽은 119개만 보면 80.7 %.
- 영향: Table I는 이 칸에 conservative-adaptive를 고르므로 값은 그대로.
  하지만 F3 때문에 UniVLA WidowX에는 제대로 된 fusion 결과가 사실상 없음.
  전체 sweep 짝지은 검정의 "-39.5점"은 크래시 때문에 생긴 허수.
- 어떻게: 더 큰 GPU에서 이 설정 200개만 다시 돌리거나, "UniVLA WidowX에서는
  task-aware를 실행하지 못함"으로 표기.

**F2. CronusVLA에서 이름표와 실제 실행이 다른 설정.**
- WidowX guarded reuse: strict, moderate, aggressive 세 폴더가 전부 strict
  문턱값으로 돌았고(기록된 인자가 동일) 결과도 에피소드 단위로 완전히 같음
  (200개 중 72 성공, 9,947스텝 중 재사용 5번). Table I의 CronusVLA reuse
  칸은 "strict만 돌린 값"임. Fractal은 세 preset이 제대로 달랐음.
- temporal fusion(두 환경 모두): 126개 CronusVLA 파일 어디에도 fusion 인자가
  없고, 세 설정이 성공, 스텝, 호출 수, 융합 패치 수까지 동일함. CSV에는
  keyframe 간격 1이라고 적혀 있는데, 간격 1이면 매번 새로 계산하므로
  에피소드에 기록된 "호출당 융합 패치 102~118개"와 모순됨. 멘토님이 말씀한
  재실행은 이 zip에 들어 있지 않음.
- depth pruning은 정상: DiT(행동 생성기) 12블록 기준으로 규칙이 파일에
  명시돼 있고 검증됨(WidowX 10; 8,10; 4,6,8,10 / Fractal 3,6,8,10).
- 참고: CronusVLA 파일은 seed, 호출별 지연 시간, 모델 이름을 기록하지 않음.
- 어떻게: reuse moderate/aggressive 400 에피소드와 fusion 설정 확인이
  필요함. CronusVLA는 0.5B라 빠름(에피소드당 약 6초).

**F3. conservative-adaptive fusion이 SpatialVLA와 UniVLA에서 한 번도
작동하지 않음.**
- 무슨 일: 17개 폴더 모두 keyframe 수 = 호출 수, 재사용 토큰 0. 즉 매번
  전체를 새로 계산했고 fusion이 실제로는 안 켜진 셈. UniVLA WidowX는 원래
  정책과 200개 에피소드가 완전히 같음.
- 영향: Table I의 UniVLA WidowX fusion 칸(87.50, +0.00)과 Table II의 UniVLA
  Object, Spatial fusion 칸은 사실상 "원래 정책"임. OpenVLA, CogACT,
  MiniVLA에서는 정상 작동(재사용 토큰 64개).
- 어떻게: 각주로 "이 설정은 강제 keyframe 조건 때문에 해당 백본에서
  한 번도 재사용을 하지 않았다"고 밝히면 됨. 재실행 불필요.

**F4. CogACT의 task-aware는 motion-entropy와 같은 실행.**
- 무슨 일: 84개 CogACT 파일 모두 `task_relevance_supported: false`, 즉
  텍스트-이미지 attention을 수집하지 않음. 결과가 파일 단위로 motion-entropy와
  동일.
- 영향: CogACT는 fusion 설정을 3개가 아니라 2개 시험한 것.
- 어떻게: 각주 또는 CogACT 행에서 task-aware 제외. 재실행 불필요.

**F5. SmolVLA는 두 가지 코드 버전과 다섯 종류 GPU가 섞여 있음.**
- 무슨 일: 27개 폴더(Original, foveation, action repeat, depth 일부)는 옛
  구현(transformers 4.51.3, lerobot 0.4.4), 29개 폴더(reuse 3, fusion 3,
  나머지 depth)는 새로 짠 구현이고 인자 기록이 없음. 새 구현 에피소드에는
  GPU 이름이 있는데 RTX 5090, RTX 6000 Ada, RTX PRO 6000, L40S, A6000이
  한 폴더 안에서도 섞임(호출당 약 262 / 185 / 470 ms로 제각각). 파일 자체에
  "옛 구현과 새 구현의 시간은 비교 대상이 아니며 개선 주장 없음"이라고
  적혀 있음.
- depth pruning: 층 번호가 30, 28, 24인데 SmolVLA의 VLM은 16층이고
  `calibrated: false`(영향도 계산 안 함). 지연 시간은 안 줄고(275 → 278 →
  277 ms) 성공률만 무너짐(Long 42 → 9 → 3 → 0). 우리 Methods가 말하는
  절차가 아님.
- CSV에는 fusion 재사용 토큰이 0으로 적혀 있지만 에피소드에는 3~26개로
  기록됨.
- 영향: Table II의 SmolVLA reuse, fusion, depth 2(Long, Goal), depth 4(Long,
  Goal, Object) 행의 지연 시간 변화는 비교가 성립하지 않음.
- 어떻게: SmolVLA 지연 시간 열을 빼거나, 한 GPU와 한 구현으로 Original,
  foveation, repeat, depth 1을 다시 돌리거나(4 suite x 100 = 400 x 설정 수),
  각주로 처리. depth 층 번호의 뜻은 멘토님께 물어야 함.

**F6. guarded reuse는 게이트가 거의 안 열림.**
- 무슨 일: 재사용된 스텝 비율이 최대 10 %(OpenVLA LIBERO aggressive)이고,
  UniVLA 두 환경, SpatialVLA WidowX, CronusVLA WidowX는 수천~수만 스텝 중
  0~38번뿐.
- 영향: "guarded reuse는 유의한 변화 없음"이라는 결과는 사실 "거의 실행이
  안 됐음"에 가까움. UniVLA WidowX strict와 UniVLA LIBERO Object, Spatial
  reuse 행은 성공, 스텝, 호출 수가 원래 정책과 같음.
- 어떻게: Results에 게이트 열린 비율을 같이 적으면 됨. 재실행 불필요.

**F7. 멘토님 `summary.csv`가 정한 규칙(성공률 최고, 동률이면 스텝 적은 쪽)을
안 따른 곳.**
- WidowX: CronusVLA depth pruning을 35.5 %(depth 1)로 골랐는데 depth 2가
  36.0 %. 어떤 규칙으로도 재현 안 됨.
- LIBERO: 동률 8곳을 스텝이 아니라 cycle latency가 낮은 쪽으로 고름.
- 처리: 두 표 생성 스크립트가 이제 백본별 CSV에서 규칙대로 직접 고르고,
  멘토님 summary와 다른 칸을 출력함. Table I의 CronusVLA WidowX depth 행이
  36.00 (+2.00) / 118.76 (-7.08) / 50.52 (-0.04)로 바뀌었고 나머지 179칸은
  그대로임을 재확인함.

**F8. OpenVLA LIBERO Long의 conservative-adaptive 실행이 없음.** 그래서
OpenVLA Long fusion 칸은 두 설정 중에서 고른 것(motion-entropy 49 %).

## 5. Results 쓸 때 바로 쓸 수 있는 사실

- **Latency 열의 뜻.** CSV의 cycle, policy, control_frequency_hz는 백본
  하네스마다 뜻이 다름(어떤 건 호출당, 어떤 건 스텝당, 어떤 건 chunk당).
  우리가 쓰는 per-step wall-clock(에피소드 시간 / 스텝)만 모든 백본에서
  뜻이 같고, 에피소드 파일에서 정확히 재계산됨.
- **가족별 지연 시간.** foveation은 어디서나 스텝당 2~17 % 느려짐(블러
  비용). action repeat는 k = 2에서 0.53~0.73배, k = 4에서 0.28~0.59배로
  1/k만큼 줄지 않음(환경 시뮬레이션 시간은 그대로). depth pruning은
  SmolVLA를 빼면 모든 백본에서 호출당 시간이 단조 감소(예산 4에서 0.87~0.95
  배). task-aware는 UniVLA, MiniVLA, OpenVLA에서 호출당 7~20 % 느려지고
  CogACT, SpatialVLA에서는 안 느려짐. motion-entropy와 conservative-adaptive는
  어디서나 2 % 이내.
- **Avg. Steps는 성공률을 따라감.** 실패는 전부 상한까지 돌기 때문에 평균
  스텝은 "성공한 에피소드 길이와 상한의 성공률 가중 평균"임. `truncated`
  플래그는 백본마다 뜻이 달라 쓸 수 없음(CronusVLA, SmolVLA는 없음, LIBERO는
  항상 false).
- **하락이 어디서 오는지.** action repeat는 모든 과제에서 고르게 무너짐
  (UniVLA WidowX 42, 49, 44, 40 → 12, 3, 6, 4). depth pruning 하락은 특정
  과제에 몰림(MiniVLA 예산 1: 큐브 쌓기 36 → 8, 나머지는 거의 그대로;
  SpatialVLA 예산 4: 가지 50 → 8, 당근은 13 그대로). OpenVLA foveation도
  한두 과제에 몰림(WidowX 가지 37 → 8, Fractal move_near 31 → 3).
- **keep 20 대 keep 50.** Fractal과 LIBERO 전부, 그리고 MiniVLA와 OpenVLA
  WidowX에서는 keep 50이 나음. WidowX의 CogACT는 동률(105 대 105),
  CronusVLA(73 대 68), SpatialVLA(100 대 89), UniVLA(158 대 146)는 keep 20이
  앞섬.
- **유의한 상승(짝지은 검정).** CogACT WidowX depth 2(50 → 60 %, p =
  0.009), CogACT WidowX motion-entropy fusion(50 → 58 %, p = 0.020), OpenVLA
  Fractal task-aware fusion(+6.8점, p = 0.012), UniVLA Goal foveation keep
  50(+7, p = 0.039). 그 밖의 표 칸 상승은 유의하지 않음.
- **에피소드별 지연 시간은 안정적.** CogACT, MiniVLA, OpenVLA, SpatialVLA는
  10~90 백분위 폭이 중앙값의 0.3~1.5 %, UniVLA는 9~17 %. 이상치 에피소드
  없음.

## 6. 재실행이 얼마나 필요한가

걱정하실 만한 부분이라 따로 정리합니다. 전체 47,500 에피소드 중 다시 돌려야
설득력이 생기는 것은 아래 세 가지, 최대 1,800 에피소드(3.8 %)입니다.

| 우선순위 | 무엇을 | 에피소드 수 | 대안 |
|---|---|---|---|
| 1 | UniVLA WidowX task-aware(F1) | 200 | 큰 GPU가 없으면 "실행 불가"로 표기 |
| 2 | CronusVLA WidowX reuse moderate, aggressive(F2) | 400 | 표에 "strict"로 각주 |
| 3 | CronusVLA fusion 3설정 x 2환경(F2), 설정 확인 후 | 최대 1,200 | 설정값을 알면 각주로 가능 |

각주로 처리해도 되는 것: F3(conservative-adaptive 미작동), F4(CogACT
task-aware = motion-entropy), F5의 지연 시간(SmolVLA 지연 시간 열 제외),
F8(OpenVLA Long 한 설정 없음). F5의 depth 층 번호는 뜻을 물어봐야 하고,
답에 따라 SmolVLA depth 행을 각주 처리하거나 뺄 수 있습니다.

## 7. 멘토님께 남은 질문 (한글)

1. UniVLA WidowX task-aware: 200개 중 81개가 CUDA OOM으로 죽어 실패로
   집계됨(48.0 %). 큰 GPU에서 재실행 가능한지, 아니면 "실행 못 함"으로 표기할지.
2. CronusVLA WidowX guarded reuse: moderate, aggressive 폴더가 strict
   문턱값으로 기록돼 있고 결과가 동일함. 두 설정 재실행 가능한지.
3. CronusVLA temporal fusion: fusion 인자가 기록에 없고 세 설정 결과가
   동일하며 CSV는 keyframe 간격 1. 실제로 어떤 간격과 비율로 돌았는지, 재실행이
   오는지.
4. CogACT task-aware: attention을 수집하지 않아 motion-entropy와 같은 실행임.
   CogACT에서 이 설정을 빼거나 각주로 할지.
5. conservative-adaptive가 SpatialVLA, UniVLA에서 한 번도 재사용을 안 함
   (keyframe = 호출 수, 재사용 토큰 0). 예상된 동작인지, 각주로 둘지.
6. SmolVLA: (a) 29개 폴더가 다른 구현과 다섯 종류 GPU라 지연 시간 비교가
   안 됨. 한 GPU, 한 구현으로 Original 등 재실행 가능한지, 아니면 SmolVLA
   지연 시간을 표에서 뺄지. (b) depth 층 번호 30; 28,30; 24,26,28,30이 16층
   VLM 범위를 넘고 calibrated: false임. 무엇을 가리키는 번호인지, 영향도
   계산을 했는지.
7. 표 선택: WidowX summary가 CronusVLA depth 1(35.5 %)을 depth 2(36.0 %)
   대신 골랐고, LIBERO summary는 동률 8곳을 cycle latency로 깼음. 우리는
   "성공률 최고, 동률이면 스텝 적은 쪽"으로 백본별 CSV에서 고름. 확인 요청.
8. OpenVLA LIBERO Long conservative-adaptive가 없음. 실행 예정인지.
9. Setup용: attention backend가 SpatialVLA, UniVLA에만 기록됨. torch
   버전은 어디에도 없음. SmolVLA 옛 구현만 transformers 4.51.3, lerobot 0.4.4.
   GPU는 SmolVLA 새 구현에만 기록됨.

## 8. 파일 위치

- 영문 원본: `experiments/paper/DataAudit_2026-09-11.md`
- 스크립트, 결과, 네 개 점검 보고서, 파일 목록(`file_manifest.csv`):
  `experiments/data_audit_2026-09-11/`
- 과제별 성공 표(2,286행): `experiments/data_audit_2026-09-11/per_task_success.csv`
- 짝지은 검정: `experiments/paper/PairedResults.md`(표 칸 110개),
  `PairedResultsAll.md`(전체 285 설정)
- 표 생성: `experiments/make_simpler_table.py`, `make_libero_table.py`
