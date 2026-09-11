# IV-B Results and Discussion, IV-C Analysis 초안 해설 (한글)

`results.tex`의 문단별 뜻과 근거입니다. Overleaf에 붙일 때 이 문서를 옆에
두고 읽으면 각 문장이 어디서 왔는지 알 수 있습니다. 숫자는 전부
`tablesimpler.tex`, `tablelibero.tex`, `PairedResults.md`, `PairedResultsAll.md`,
`data_audit_2026-09-11/per_task_success.csv`에서 왔고, 세 번의 독립 검증(숫자
재계산, 문체와 I~III절 용어 대조, 심사위원 관점 공격)을 거친 두 번째 판입니다.

## 쓰기 전에 정한 원칙

- 각주 없음. 예외는 전부 본문 문장으로 넣음.
- 상승 4칸은 "significant"라고 부르지 않고 "nominal(명목상)"이라고 부름.
  110칸 각각이 2~3개 설정 중 최고를 고른 것이라 사후 선택된 비교이고,
  다중비교 보정(Benjamini-Hochberg, Bonferroni)을 걸면 상승은 하나도 안 남기
  때문. 이렇게 써야 심사위원이 "그 +9.5는 우연 아니냐"고 할 때 우리가 먼저
  말해 둔 것이 됨.
- 유의하지 않은 변화에는 "lowers, removes, adds" 같은 동사를 쓰지 않고
  "changes by"로 씀.
- 재실행에 걸린 문장은 `%PENDING` 주석으로 표시. 답이 오면 그 문장만 고침.

## IV-B 문단별

**도입 문단.** 표 두 개가 무엇을 보여주는지(dense 정책과 가족별 최고 설정,
괄호는 dense 대비 변화, Latency는 스텝당 벽시계 시간), 그리고 검정 결과
요약. 110칸 중 26칸이 보정 전 p < 0.05, 그중 22칸 하락. 하락 중 16칸은 BH
보정, 11칸은 Bonferroni 보정을 통과. 상승 4칸(CogACT WidowX depth +9.5,
CogACT WidowX motion-entropy fusion +8.0, OpenVLA Fractal task-aware +6.8,
UniVLA Goal foveation +7.0)은 보정을 통과하지 못하므로 "재현해야 할 관찰"로
보고. 마지막 문장은 방어용입니다. 개입이 한 번도 작동하지 않은 4칸은 dense와
에피소드 단위로 완전히 같고, 작동한 스텝이 없는 다른 2칸은 100개와 200개 중
1개, 2개 에피소드만 다르므로 실행 간 잡음은 100개 중 1개 수준이고, "seed
하나"의 한계는 측정 잡음이 아니라 다른 초기 상태로의 일반화 문제라는
뜻입니다.

**Cells that are not what their row says (행 이름과 실행이 다른 칸).**
감사에서 찾은 예외를 한 문단에 모아 한 번만 말합니다. UniVLA WidowX와 Spatial
fusion 칸은 설정이 한 번도 작동하지 않아 dense와 같음. UniVLA reuse 5칸은
재사용이 0~5번뿐이라 한 에피소드만 빼고 dense와 같음. SpatialVLA WidowX reuse
칸은 재사용 38번에 에피소드 3개 차이. CronusVLA WidowX reuse 칸은 재사용
5번인데 에피소드 14개가 달라서, seed를 기록하지 않는 그 하네스는 결정적이지
않음(이 사실은 CronusVLA의 작은 변화량을 읽을 때 잡음 수준으로 쓰임).
CronusVLA fusion 칸은 설정 기록이 없는 한 번의 실행, CronusVLA WidowX reuse
칸은 strict만 실행. UniVLA WidowX task-aware는 81개 크래시로 모든 집계에서
제외. SmolVLA의 reuse, fusion, 높은 depth 예산은 다른 구현과 GPU라 지연 시간을
읽지 않고 성공률 변화도 짝지어진 실행이 아님, depth 층 번호는 고정값. OpenVLA
Long fusion 칸은 두 설정 중 선택. 재실행 결과가 오면 이 문단이 줄어듭니다.

**Action repeat.** k = 2는 호출 수를 반으로 줄이고 스텝당 시간을 dense의
0.52~0.72배로 만듦(정확히 반이 안 되는 이유는 IV-C). WidowX에서는 모든
백본이 떨어지는데 OpenVLA, CronusVLA, CogACT, UniVLA는 15.5~75.0점(p <
0.001), SpatialVLA와 MiniVLA는 4.0, 6.0점(유의하지 않음). Fractal에서는 같은
설정이 −1.2~+4.4로 유의한 변화 없음. 그래서 CogACT, OpenVLA, CronusVLA에서
"WidowX에서는 실패, Fractal에서는 통과"라는 Intro 문장이 성립. LIBERO에서
UniVLA는 모든 suite에서 58~75점 하락, OpenVLA와 SmolVLA는 −3~−12(유의하지
않음). UniVLA가 크게 무너지는 이유는 한 호출에 5개(WidowX) 또는 10개(LIBERO)
행동을 내보내는 chunk 구조라 한 번 반복하면 10~20스텝 동안 피드백이 없기
때문. k = 4는 22칸 중 20칸에서 유의하게 하락.

**Foveation.** 스텝당 시간이 모든 칸에서 오르고(UniVLA 빼고 3~16 %, UniVLA는
chunk 때문에 블러 비용이 나뉘어 작음). 성공률은 OpenVLA(WidowX −13.5,
Fractal −11.2, Long −12.0), SmolVLA 세 suite(−13~−21), UniVLA WidowX(−8.5)에서
유의하게 하락하고 UniVLA Goal(+7.0)에서 한 번 명목상 상승. CogACT, CronusVLA,
SpatialVLA, MiniVLA는 모든 환경에서 5점 이내. keep 0.2 대 0.5 비교는
OpenVLA에서 0.2가 23~28점 빠지는 곳에서 0.5는 12점 이하이고, WidowX의 CogACT,
CronusVLA, SpatialVLA, UniVLA에서는 0.2가 오히려 같거나 최대 6점 앞섬.

**Depth pruning.** 호출당 시간은 SmolVLA를 빼고 모든 백본에서 예산에 따라
단조 감소, 스텝당 시간은 표에서 1~6 % 감소. 성공률은 이 연구에서 가장 뚜렷한
부호 역전: CogACT WidowX는 예산 2에서 50.0 → 59.5(p = 0.009, 보정 전)인데
CogACT Fractal에서는 어느 예산도 1.6점 이상 안 움직임. 같은 WidowX 과제에서
MiniVLA는 예산 1에서 −17.5(p < 0.001), 예산 2와 4에서 36점 전부 잃음.
SpatialVLA는 −6.5, −17.0, −31.0. OpenVLA는 SimplerEnv에서 +4.0, +5.2(유의하지
않음), LIBERO Spatial −16.0(p = 0.011), 예산 4에서는 모든 suite에서 −22~−34.
CronusVLA(자르는 것은 12층 행동 디코더)는 +2.0, +0.8, 예산 4에서 −4.4, −5.5.
UniVLA −5.5(p = 0.052), LIBERO +1~+3. SmolVLA 예산 1에서 −14~−33. 마지막
문장: 같은 예산이 32층(CogACT), 26층(SpatialVLA), 24층(MiniVLA)에서 다른
비율이고 고르는 층도 다르므로 예산만으로는 부호를 예측할 수 없다.

**Guarded reuse.** 유의한 칸이 없고 모든 칸이 5점 이내. 게이트 기록이 그
이유를 "한정"함(설명이 아니라 상한): 재사용된 스텝이 OpenVLA LIBERO
aggressive에서 10 %, OpenVLA 나머지와 MiniVLA에서 2~6 %, 다른 백본은 5 % 미만,
UniVLA는 천 스텝에 한 번 미만, SpatialVLA WidowX와 CronusVLA WidowX는
9,000~9,950스텝 중 2~38번. 스텝당 시간은 SimplerEnv에서 4 % 미만 변화, OpenVLA
LIBERO에서 2~10 % 감소로 건너뛴 스텝에 비례. preset을 느슨하게 하면 재사용
비율이 1.3~4.5배 늘지만(재사용이 1 % 넘는 곳 기준) 성공률은 7점 이내로 유의한
변화 없음.

**Temporal fusion.** motion-entropy와 conservative-adaptive는 호출당 시간을
3.5 % 이내로 바꾸고, task-aware의 attention 수집은 OpenVLA 7~9 %, MiniVLA
14 %, UniVLA LIBERO 10~16 %를 더함. 명목상 상승은 CogACT WidowX
motion-entropy(+8.0, p = 0.020)와 OpenVLA Fractal task-aware(+6.8, p =
0.012), 그리고 p > 0.1로 +7.0인 OpenVLA WidowX와 Spatial. 나머지는 −7.0~+6.0으로
유의하지 않음. CogACT는 선택기가 도는 지점에서 텍스트-이미지 attention을 내주지
않아 task-aware가 motion-entropy와 같아지므로 설정이 둘. SpatialVLA와
UniVLA에서 conservative-adaptive는 재사용 패치를 하나도 안 골라 dense와 같고,
그래서 SpatialVLA 칸과 UniVLA Long, Goal, Object 칸은 다른 두 설정에서 왔으며
UniVLA WidowX motion-entropy는 −5.5(p = 0.035). OpenVLA WidowX에서는 설정에
따라 +7.0(task-aware)과 −2.5(motion-entropy)로 9.5점 차이지만 둘 다 dense와
유의하게 다르지 않음. fusion은 시간이 아니라 정확도를 위한 후보인데, 실행된
20칸 중 2칸에서 보정 전 p < 0.05, 보정 후에는 0칸.

**Across families.** 세 가지 패턴. (1) 부호는 규칙의 속성이 아니다: 같은
WidowX 과제에서 depth pruning이 CogACT +9.5, MiniVLA −17.5. 같은 Goal
suite에서 foveation이 UniVLA +7.0, SmolVLA −13.0(상승 둘은 보정 전). (2)
부호는 백본의 속성도 아니다: action repeat가 CogACT WidowX −38.0, Fractal
−1.2. keep 0.5가 UniVLA WidowX −14.5(p < 0.001), Goal +7.0. (3) 지연 시간과
성공률은 같이 움직이지 않는다: foveation은 모든 백본에서 느리고 셋에서
나쁘며, depth pruning은 여섯 백본에서 빠르고 넷에서 나쁘고, action repeat는
가장 빠르고 가장 나쁨. 빠르면서 명목상 좋아진 유일한 칸(CogACT WidowX
depth)은 보정을 못 넘고 Fractal에서 0.0. "이 칸들을 근거로 어떤 설정도
추천하지 않는다"로 닫음.

## IV-C 문단별

**Episode length follows success.** 크래시 아닌 실패는 전부 스텝 상한(WidowX
60 또는 120, Fractal 80, LIBERO 220~520)까지 가므로 평균 스텝은 "성공한
에피소드 길이와 상한의 성공률 가중 평균"이고, 따라서 Avg. Steps 열은 Success
열을 따라가며 독립적인 효율 지표가 아님. 예외는 CronusVLA Fractal로, 14개 실행을
합쳐 성공한 에피소드 121개도 상한에서 끝남.

**What a repeated call saves.** action repeat의 스텝당 시간은 k = 2에서
0.52~0.72배, k = 4에서 0.28~0.58배로, 호출 수만 보면 0.50, 0.25여야 하는데 그
만큼 안 줄어드는 이유: 시뮬레이터 스텝은 매 스텝 들고, UniVLA를 뺀 SimplerEnv
하네스에서는 호출이 여러 스텝을 품기 때문에 호출당 시간이 k = 4에서 120~230
ms 늘어남. 모델 계산만 호출 수에 비례. 같은 계산이 guarded reuse에도 적용되어
절약은 "재사용 비율 × 호출 시간"이고 최대 약 10 %.

**Where the drops sit.** 에피소드 기록으로 하락의 두 유형을 구분. action
repeat는 한 백본의 모든 과제에서 동시에 떨어짐(UniVLA WidowX 42, 49, 44, 40
→ 12, 3, 6, 4). depth pruning과 foveation은 특정 과제에서 떨어짐: MiniVLA
예산 1은 stack cube(36 → 8, 나머지 셋은 6 이내), SpatialVLA 예산 4는 eggplant
(50 → 8, carrot은 그대로), OpenVLA WidowX foveation은 eggplant(37 → 12, spoon은
19 → 24로 오히려 상승).

**Where the signal admits it.** 실행 중 신호를 쓰는 두 후보(reuse, fusion)는
신호가 있을 때만 작동함. guarded reuse는 안정된 이미지, 일치하는 두 행동,
바닥값 이상의 이동이 같은 스텝에 있어야 하는데 그런 스텝이 최대 11 %.
conservative-adaptive는 SpatialVLA와 UniVLA에서 재사용 패치를 못 찾음.
task-aware는 CogACT에서 attention을 못 찾음. 그래서 후보는 효과보다 먼저
신호에 의해 한정되고, 신호가 한 번도 안 켜진 백본은 개입 이름 아래 dense
정책을 보고하게 됨. 에피소드를 짝짓고 스텝별 게이트 기록을 남기는 프로토콜이
이것을 드러내고, 집계 성공률만 보고하는 프로토콜은 드러내지 못함. 이 마지막
문장이 논문의 프로토콜 기여와 이어짐.

## 다른 절에서 맞춰야 할 것

- **Intro**: "[six] VLA backbones"는 일곱, "[the finalized set of
  environments]"는 세 환경. "the same intervention can improve one backbone
  and degrade another"는 상승이 보정 전이므로 "leave one backbone unchanged or
  nominally better and degrade another"로. 기여 2의 "with a fixed margin for
  each quantity"와 RW 마지막 문단의 "preregistered margin that Section IV-A
  fixes"는 Results가 margin을 쓰지 않으므로, IV-A에서 margin을 정해 Results에
  한 문장을 넣거나 Intro와 RW에서 margin 문구를 빼야 함. 추천은 후자(짝지은
  검정과 보정이 이미 그 역할을 함).
- **Methods III-E** "Three settings are evaluated": CogACT는 둘, CronusVLA는
  기록 없음. Results가 본문에서 밝히므로 Methods는 그대로 두어도 됨.
- **Setup IV-A(멘토님)**: 선택 규칙(최고 성공률, 동률이면 스텝 적은 쪽),
  McNemar 검정, 스텝당 벽시계 시간 정의, 백본별 chunk 크기(UniVLA 5/10)와
  디코더 층 수(CogACT 32, SpatialVLA 26, MiniVLA 24, CronusVLA DiT 12), SmolVLA
  두 구현체가 들어가야 Results의 문장들이 근거를 가짐.
- **분량**: 현재 약 1,870단어. 표 두 개가 약 한 쪽을 차지하므로 IV-B와
  IV-C에 두 쪽 정도가 남는다면 맞고, 그보다 적으면 "Cells that are not what
  their row says"가 재실행 후 줄어드는 것과 별개로 depth pruning 문단의 예산
  4 수치, foveation 문단의 keep 0.2 문장, IV-C의 "Where the drops sit"부터
  줄이면 됨.
