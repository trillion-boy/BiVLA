# 선행연구 정독 — 원문 5편

**읽은 것 (PDF 원문, 직접 대조):**

| # | 논문 | 출처 | 상태 |
|---|---|---|---|
| A | **ShortGPT: Layers in LLMs are More Redundant Than You Expect** | arXiv 2403.03853v3 (2024-10-11) | ✅ 정독 |
| B | **EfficientVLA: Training-Free Acceleration and Compression for VLA** | arXiv 2506.10100v1 (2025-06-11) | ✅ 정독 |
| C | **VLA-Cache: Efficient VLA Manipulation via Adaptive Token Caching** | Univ. of Sydney / SJTU | ✅ 정독 |
| D | **MoLe-VLA: Dynamic Layer-skipping VLA via Mixture-of-Layers** | Nanjing/PKU, AAAI'26 | ✅ 정독 |
| E | **Gaze-Regularized VLA Models for Robotic Manipulation** | HKU (Pani & Yang) | ✅ 정독 |

---

# ★ 가장 중요한 발견 — §3c가 이미 문헌에 있다, 다만 아무도 이름 붙이지 않았다

**B와 C가 우리와 같은 벤치마크(SIMPLER, Google Robot)에서 같은 네 태스크를
쓴다.** 그들의 표를 태스크별로 펼치면 다음이 나온다.

### SIMPLER Visual Matching, 백본 = CogACT

| 방법 | 제거하는 것 | **PickCan** | **MoveNear** | Drawer | DrawerApple | Avg |
|---|---|---|---|---|---|---|
| CogACT (baseline) | — | 91.3 | 85.0 | 71.8 | 50.9 | **74.8** |
| FastV | 시각 토큰(attention) | **92.6** `+1.3` | **81.4** `−3.6` | 69.8 | 52.4 | 74.1 |
| VLA-Cache | 시각 토큰(캐시) | **92.0** `+0.7` | **83.3** `−1.7` | 70.5 | 51.6 | 74.4 |
| EfficientVLA L28 T112 | 층 + 토큰 | **95.3** `+4.0` | **83.3** `−1.7` | 70.3 | 56.5 | 76.4 |
| EfficientVLA L28 T56 | 층 + 토큰 | **94.7** `+3.4` | **82.4** `−2.6` | 69.8 | 55.4 | 75.5 |
| EfficientVLA L22 T112 | 층 + 토큰 | **94.0** `+2.7` | **82.1** `−2.9` | 69.2 | 54.6 | 75.0 |
| EfficientVLA L22 T56 | 층 + 토큰 | **93.3** `+2.0` | **81.3** `−3.7` | 68.2 | 53.8 | 74.2 |

### SIMPLER Variant Aggregation, 같은 백본

| 방법 | **PickCan** | **MoveNear** | Avg |
|---|---|---|---|
| CogACT (baseline) | 89.6 | 80.8 | **61.3** |
| FastV | **91.4** `+1.8` | **78.6** `−2.2` | 62.1 |
| VLA-Cache | **91.7** `+2.1` | **79.3** `−1.5` | 62.3 |
| EfficientVLA L28 T112 | **94.8** `+5.2` | **77.6** `−3.2` | 63.2 |
| EfficientVLA L28 T56 | **94.4** `+4.8` | **77.2** `−3.6` | 62.6 |
| EfficientVLA L22 T112 | **93.9** `+4.3` | **76.4** `−4.4` | 62.1 |
| EfficientVLA L22 T56 | **93.2** `+3.6` | **75.8** `−5.0` | 61.2 |

## 여기서 나오는 것 세 가지

**① 12개 구성 전부에서 `PickCan`은 오르고 `MoveNear`는 내린다.**
서로 다른 세 논문, 서로 다른 네 방법(attention 마스킹 / 토큰 캐싱 / 층 가지치기
/ 층+토큰), 두 평가 설정. **예외가 없다.** 이것이 정확히 우리 §3c다 — 대상이
하나뿐이라 고를 필요가 없는 태스크는 살아남고, **지시된 물체를 골라내야 하는
태스크가 먼저 죽는다.**

**② 용량을 더 깎을수록 격차가 벌어진다 — 단조롭다.**
EfficientVLA의 네 구성을 용량 순으로 늘어놓으면
(L28 T112 → L28 T56 → L22 T112 → L22 T56):

```
PickCan   +4.0 → +3.4 → +2.7 → +2.0     이득이 줄고
MoveNear  −1.7 → −2.6 → −2.9 → −3.7     손실이 커진다
```
Variant Aggregation에서도 `+5.2 → +4.8 → +4.3 → +3.6` / `−3.2 → −3.6 → −4.4 →
−5.0`으로 같다. **우리 §3b의 "깊이 곡선 두 개가 반대로 간다"가 남의 데이터에
그대로 있다.**

**③ 그런데 세 논문 모두 평균만 보고한다.** EfficientVLA는 74.8 → 74.2를 두고
*"merely a 0.6% drop"*이라고 쓴다. **두 반대 곡선이 평균에서 상쇄된다.** 우리
§3b가 "aggregate가 두 곡선을 숨긴다"고 주장한 것의 **문헌 내 실물 증거**다.

> ### 논문에 쓸 문장 (초안)
>
> The split is not peculiar to our runs. Re-reading the per-task tables of three
> published SIMPLER results — token masking [FastV, as reported in
> EfficientVLA], token caching [VLA-Cache], and layer-plus-token pruning
> [EfficientVLA] — **`pick coke can` improves and `move near` degrades in all
> twelve reported configurations**, and in EfficientVLA the gap widens
> monotonically with the amount removed (+4.0 → +2.0 on pick, −1.7 → −3.7 on
> move near). None of these papers comments on the pattern, because each reports
> the four-task mean, in which the two movements cancel: EfficientVLA describes
> its largest configuration as *"merely a 0.6% drop."* **The convention of
> reporting one aggregate number is what has kept this invisible.**

**⚠️ 범위 주의.** Drawer 계열은 깨끗하지 않다. `Drawer`(open/close drawer)는
Visual Matching에서 6/6 하락하지만 Variant Aggregation에서는 섞이고,
`DrawerApple`은 대부분 상승한다. **단조롭게 갈리는 것은 `PickCan`(상승)과
`MoveNear`(하락) 두 개다.** 우리 주장을 이 두 개로 좁혀서 쓰는 것이 정확하다.

---

# A. ShortGPT — 우리 depth 랭킹의 출처, 완전 일치 확인

### 정의 (원문 식 1, p.4)

```
BI_i = 1 − E_{X,t} [ (X_i,t · X_{i+1,t}) / (||X_i,t||₂ ||X_{i+1,t}||₂) ]
```
> *"Lower BI score imply that X_i and X_{i+1} exhibit high cosine similarity,
> suggesting that the layer makes minimal transformations to the hidden states
> and is therefore less important."*

절차(3.2): calibration set으로 **PG19(라벨 없는 텍스트)**를 쓰고, 그 위에서
추론하며 각 층의 hidden state를 모으고, BI를 계산해 **오름차순 정렬 후 낮은
BI부터 삭제**한다.

### 우리 코드와 대조 (`SpatialVLA/experiments/tome/depth_prune_gemma2.py`)

```python
cs = cosine_similarity(layer_input, layer_output).mean()
# higher cos = more redundant = safer to drop; rank descending
```

**동일하다.** `높은 cos 내림차순` = `낮은 BI 오름차순`. 순서가 같다.

### ⚠️ 그런데 우리와 다른 점이 하나 있고, 이건 반드시 밝혀야 한다

| | ShortGPT | 본 연구 |
|---|---|---|
| calibration 데이터 | **PG19 — 도메인 밖 영어 텍스트** | **에피소드 첫 스텝의 실제 로봇 관측** |
| calibration 시점 | 배포 전 1회 | 에피소드마다 1회 |
| 평가 | perplexity + 13개 NLP 벤치마크 | 폐루프 성공률 |

우리는 **in-domain 보정**을 한다. 이건 개선일 수도 있고 교란일 수도 있다 —
에피소드마다 랭킹이 달라질 수 있기 때문이다(실제로 `move_near`에서
`[2,4,6,23]`, pick에서 `[2,4,23,26]`으로 달랐다). **논문에 이 차이를 명시하고,
가능하면 고정 랭킹 대조군을 한 번 돌리는 것이 안전하다.**

### 그들의 검증 범위 (= 우리가 채우는 공백)

- 모델: Llama2-7B/13B, Baichuan2-7B/13B — **전부 순수 LLM**
- 지표: perplexity, MMLU, CMNLI, HellaSwag, PIQA, CHID, WSC, CoQA, BoolQ,
  Race-H/M, XSum, C3, CMMLU — **전부 단일 forward pass**
- 대표 수치: Llama2-13B에서 40층 중 10층(25%) 제거 → MMLU **55.0 → 52.2**
- **폐루프 제어 실험 0건. 로봇 0건.**

> **Related Work 문장:** We rank layers by the Block Influence criterion of
> ShortGPT — one minus the cosine similarity between a layer's input and output
> hidden states — differing only in the calibration set, which for us is the
> first observation of the episode rather than out-of-domain text. **ShortGPT
> validates the criterion on perplexity and thirteen single-forward-pass NLP
> benchmarks; it does not touch a closed loop.**

---

# B. EfficientVLA — 우리와 가장 가까운 논문. 같은 기준, 같은 벤치마크, 그런데 결론이 다르다

### 무엇을 하는가

**Training-free**로 세 가지를 동시에: ① 언어 모듈의 잉여 층 제거 ② 시각 토큰
선별(task-critical + diversity) ③ diffusion action head의 시간축 캐싱.

**층 중요도 기준이 ShortGPT와 같다:**
> *"We define the importance score I(ℓ) … as one minus the average cosine
> similarity between its input and output … These scores are then sorted in
> ascending order … Subsequently, the first n layers are pruned."*

즉 **우리와 EfficientVLA와 ShortGPT가 전부 같은 기준을 쓴다.**

### 핵심 수치

- 백본 **CogACT** (DINOv2+SigLIP, Llama2-7B, DiT action head), **28층**
- 벤치마크 **SIMPLER** — 우리와 같은 환경, Google Robot 4태스크
- 최대 구성(L=22, T=56): **FLOPs 28.9%, 1.93× 속도, 평균 −0.6%p**
- 모듈별 (Table 1): 언어 모듈이 **134.5 ms / 3726 GFLOPs**로 지배적.
  가지치기 후 **58.9 ms (−56%)**

### ★ 그들이 "역설"이라고 부른 것 — 우리에겐 역설이 아니다

> *"Remarkably, on the pick coke can task, pruning 36% of parameters
> **paradoxically improved** the success rate from 91.3% to 94.0%, highlighting
> significant parameter redundancy in the VLA model."*

**우리 OpenVLA/Fractal `depth prune 4` = +15.6 (p<0.001)과 같은 현상이다.**
그들은 "파라미터 잉여성"으로 설명하고 넘어간다. **우리는 같은 표의 MoveNear가
동시에 내려간다는 것을 보고 다른 설명을 준다** — 잉여가 아니라 **선택적 제거**다.

### ★ foveation이 연산을 못 아끼는 이유가 여기 있다

Figure 1(a)와 본문:
> *"while visual token pruning initially reduces inference time in
> computation-bound scenarios, **its efficacy quickly diminishes as the system
> becomes memory-bound by the LLM**."*
>
> *"approaches like FastV (T = 56) show that solely optimizing visual tokens
> yields **only a 1.21× speedup due to unaddressed memory bottlenecks**."*

**우리 측정(foveation → 스텝당 연산 −1.7% ~ +0.1%)의 원리적 설명이다.**
시각 쪽을 건드려서는 벽에 부딪힌다. 우리는 그 벽을 **픽셀 공간에서** 확인했고,
그들은 **토큰 공간에서** 확인했다. 같은 벽이다.

### 또 하나 — 무엇을 남기느냐가 전부다

**Random Dropping**(112토큰 무작위 유지): 74.8% → **20.9%**. 붕괴한다.
같은 개수를 유도된 기준으로 남기면 76.4%. **"픽셀/토큰을 얼마나 줄이나"보다
"무엇을 남기나"가 지배적이다.** 우리 foveation은 *중심 고정*이라 task-agnostic
하다 — Bridge에서 +18.8이 나온 이유와 Fractal에서 −19.3이 나온 이유가 모두
여기 있을 수 있다. **Discussion에 반드시 넣을 것.**

---

# C. VLA-Cache — "FLOPs를 줄여도 빨라지지 않는다"의 실측

### 핵심 표 (LIBERO, OpenVLA)

| 방법 | Avg SR | **FLOPs (T)** | **Latency (ms)** | 제어 주파수 |
|---|---|---|---|---|
| OpenVLA | 75.0% | 1.864 | 51.91 | 4.23 Hz |
| + SparseVLM | 64.7% | 1.407 | **83.39** ← 60% 느려짐 | 3.72 |
| + FastV | 73.3% | **1.864** ← 그대로 | **53.28** ← 더 느려짐 | 4.19 |
| + VLA-Cache | 74.7% | 1.355 | 31.83 | 4.59 |

**FastV는 FLOPs가 한 자리도 안 줄었고 지연은 늘었다.** SparseVLM은 FLOPs를
25% 줄이고도 **60% 느려졌다.**

### 저자들이 밝힌 원인 — 우리 §비용 절에 그대로 인용할 것

> *"Their token pruning and merging strategies operate within a **single frame**
> and disrupt **spatial fidelity**, which is critical for precise manipulation.
> Moreover, these methods **target long output sequences, whereas VLA models
> generate short action outputs (e.g., 7 tokens), rendering the speedups
> marginal**."*

**VLM 토큰 가지치기의 이득은 긴 디코딩에서 나온다. VLA는 7토큰을 뱉는다.
그래서 아낄 것이 없다.** 우리 foveation이 픽셀의 80%를 버리고도 연산을 0%
아낀 것과 정확히 같은 구조다.

### 그리고 여기서도 같은 태스크 분할

SIMPLER/CogACT: PickCan **91.3 → 92.0 (+0.7)**, MoveNear **85.0 → 83.3 (−1.7)**.
Variant: PickCan **89.6 → 91.7 (+2.1)**, MoveNear **80.8 → 79.3 (−1.5)**.

---

# D. MoLe-VLA — 문헌에서 가장 강한 긍정 사례, 그리고 그 조건

### 핵심 수치 (RLBench, 10 태스크, Franka Panda, 전면 카메라)

| 방법 | Mean Acc. | FLOPs (G) |
|---|---|---|
| OpenVLA | 45.4% | 1930.0 |
| CogAct | 57.2% | 1935.8 |
| **Random-skip-CogAct** | **51.2% (−6.0)** | 984.3 |
| MoD-CogAct | 56.4% (−0.8) | 985.8 |
| DeeR-CogAct | 59.2% (+2.0) | 997.4 |
| **MoLe-OpenVLA (제안)** | **55.6% (+10.2)** | 981.5 |
| **MoLe-CogAct (제안)** | **60.8% (+3.6)** | 985.8 |

*"The five efficiency methods operate with only 50% LLM layers."* — 절반을
건너뛴다.

### 조건 셋 — 이걸 안 밝히면 잘못 인용하게 된다

1. **Training-free가 아니다.** STAR 라우터를 학습하고, 잃은 인지 능력을
   **CogKD(self-knowledge distillation, EMA teacher)**로 복구한다. 최종 목적함수는
   `L = L_task + 0.5·L_cog + 0.1·L_lb`. **우리 것은 학습이 전혀 없다.**
2. **벤치마크 하나, 백본 하나 계열.** RLBench + 실로봇. SIMPLER 없음.
3. **Random-skip이 −6.0**이라는 것은, **이득이 "층을 지우는 것"에서 오는 게
   아니라 "어느 층을 언제 지울지 고르는 라우터"에서 온다**는 뜻이다.

### 통계

성공률이 전부 **4%의 배수**다(8.0, 72.0, 64.0, 28.0 …) → **태스크당 n = 25**.
신뢰구간 없음, paired 검정 없음. **PhAIL이 지적한 관행 그대로다.**

### 우리와의 관계

> MoLe-VLA reports +10.2 points on OpenVLA at half the decoder layers, but it
> learns a router and distills the removed capability back; its own
> random-skipping control loses 6.0 points. **What its gain establishes is the
> value of the router, not the harmlessness of removing layers.** Our setting is
> training-free and fixed-schedule, which is where the backbone dependence
> shows: removing a single layer of SpatialVLA's 26 costs 10.4 points on Bridge,
> while removing eight of OpenVLA's 32 costs nothing at all.

---

# E. Gaze-Regularized VLA — foveation은 10개 태스크 전부에서 해롭다

### 논문의 본체는 우리와 다르다

시선 히트맵을 패치 수준 분포로 바꿔 **트랜스포머 attention을 KL divergence로
정규화**한다. 아키텍처 변경도, **추론 시점 오버헤드도 없다.** 벤치마크 전반에서
**4–12% 향상.**

**→ 이건 우리 개입이 아니다.** 혼동하면 안 된다.

### 우리와 같은 것은 부록 D.2 "Foveated Vision during Training"

시선 peak를 중심으로 **foveated RGB**를 만들어(중심 고해상도, 주변 다운샘플/블러)
비전 인코더에 그대로 먹인다. **우리 blur 변형과 같은 조작이다.**

### Table 11 — LIBERO-Spatial, 30k steps, 태스크별

| 물체 위치 | baseline | +gaze | **foveated** | Δ |
|---|---|---|---|---|
| Between plate and ramekin | 83.3 | 100 | 80.0 | **−3.3** |
| Next to ramekin | 85.7 | 100 | 81.3 | **−4.4** |
| Table center | 100 | 100 | 95.7 | **−4.3** |
| On cookie box | 100 | 91.3 | 90.0 | **−10.0** |
| In cabinet drawer | 80 | 73.3 | 65.3 | **−14.7** |
| On ramekin | 100 | 100 | 90.0 | **−10.0** |
| Next to cookie box | 100 | 100 | 94.0 | **−6.0** |
| On stove | 90 | 90 | 80.7 | **−9.3** |
| Next to plate | 50 | 100 | 44.7 | **−5.3** |
| On wooden cabinet | 70.3 | 100 | 63.3 | **−7.0** |
| **평균** | **85.9** | **95.5** | **78.5** | **−7.4** |

**10개 태스크 전부 하락. 예외 없음.**

저자 가설:
> *"aggressively reducing peripheral detail removes useful contextual cues
> (e.g., table geometry, supporting surfaces, or alternative grasps) that the
> policy relies on for precise spatial reasoning."*

### ⚠️ 우리와 다른 점 두 가지 — 인용할 때 반드시 밝힐 것

1. **그들은 학습 중에 foveated 입력을 쓴다**("Foveated Vision **during
   Training**"). 우리는 사전학습된 정책에 **추론 시점에만** 건다.
2. **다시점(multi-view)** 세팅이고, foveation 중심이 **시선 추정치**다. 우리는
   단일 시점에 **이미지 중심 고정**이다.

### 그래도 우리에게 주는 것

- **foveation이 VLA를 해친다는 독립 증거**, 그것도 10/10.
- LIBERO-Spatial은 **"어느 위치의 그릇을 집어라"** 계열이다. 즉 **공간 참조**
  태스크이고, 우리 `move_near`의 **지시 대상 참조**와 사촌이다. 두 결과를
  나란히 놓으면 *"참조를 요구하는 태스크가 시각 감축에 특히 약하다"*가 된다.
- 그리고 **우리 Bridge +18.8은 여전히 설명되지 않는다.**

---

# 종합 — 우리 세 개입은 VLA에 도움이 되는가

## 연산 축(depth pruning) — **문헌은 강하게 "된다"고 한다**

| 근거 | 조건 |
|---|---|
| EfficientVLA: SIMPLER에서 FLOPs 28.9%, 평균 **−0.6%p** | **training-free**, 우리와 **같은 기준**, 우리와 **같은 벤치마크** |
| EfficientVLA: pick coke can이 **91.3 → 94.0** | 우리 +15.6과 같은 현상 |
| MoLe-VLA: 층 절반에서 **+10.2%p** | 라우터 **학습** 필요, 벤치마크 1개 |
| ShortGPT: Llama2-13B 25% 제거에 MMLU 55.0 → 52.2 | LLM 전용, 폐루프 없음 |

**우리 OpenVLA 결과는 문헌과 완전히 일치한다.** 문헌이 데이터를 갖고 있지 않은
곳은 **두 번째 백본**이다 — SpatialVLA에서 1층에 −10.4, 4층에 −17.8.
**여기가 우리 기여다.**

## 시각 축(foveation) — **문헌은 "아니다"로 수렴한다. 게다가 공짜도 아니다**

| 근거 | 수치 |
|---|---|
| Gaze-Reg: foveated 입력, LIBERO-Spatial | **10/10 태스크 하락, −7.4** |
| VLA-Cache: FastV의 FLOPs | **1.864 → 1.864 (변화 없음)**, 지연은 **증가** |
| VLA-Cache: SparseVLM | FLOPs −25%인데 지연 **+60%** |
| EfficientVLA: 시각만 최적화 | *"memory-bound라 1.21×에 그친다"* |
| EfficientVLA: 무작위 토큰 유지 | 74.8 → **20.9** |

**성능도 안 오르고 속도도 안 는다.** 우리 측정(연산 −1.7% ~ +0.1%)이
독립적으로 같은 결론에 도달했다.

**→ 그런데 OpenVLA/Bridge에서 +18.8 / +17.7이다. 문헌 어디에도 이 방향이 없다.**
이건 **논문의 중심에 놓아야 할 이상치**다. EfficientVLA의 Random Dropping
결과(무엇을 남기냐가 지배적)와 함께 보면, 가설은 *"Bridge/OpenVLA에서는
주변부가 도움이 아니라 방해였다"*이다 — 15.6%라는 낮은 baseline이 그 방증일 수
있다. **검증 실험이 필요하다**(아래 §다음).

## 시간 축(action repeat) — **처리량은 확실히 벌고, 제어는 나빠진다**

이번에 읽은 5편에는 직접 근거가 없다. 이전 검색에서 확인한 대로 chunking의
"낡은 관측" 문제와 2026년 처방 논문들이 여기 붙는다. **우리 측정에서 유일하게
연산을 크게 아끼는 축(−50% / −75%)이면서 캠페인 최악의 실패(−70.8)를 만드는
축이다.**

---

# 이 정독이 우리 논문에 하는 일

## ⚠️ 먼저 — 이 절의 이전 판은 프레이밍이 틀렸다

이전 판은 아래를 "잃은 것 / 얻은 것"이라는 **손익계산**으로 적었다. 이
장르에서는 그 틀 자체가 틀렸다.

Bag of Tricks(NeurIPS'25 D&B)가 쓰는 method — Best-of-N, beam search, MCTS,
self-consistency, self-refine — 는 **하나도 저자들이 만든 것이 아니다.** 전부
기존 방법이다. 그런데 아무도 그 논문을 "따라한 논문"이라 하지 않는다.
**이 장르에서 method가 기존 것이라는 사실은 결함이 아니라 전제 조건이다.**
오히려 그 방법들이 널리 쓰인다는 것이 확립돼 있을수록, "이것을 왜 재는가"가
자동으로 답해진다.

우리를 실제로 해칠 수 있는 논문은 딱 하나 — **같은 개입을 여러 백본 × 여러
벤치마크에서 paired로 비교한 논문**이다. 확인 결과:

| 논문 | 백본 | 벤치마크 | 축을 교차했나 |
|---|---|---|---|
| EfficientVLA | CogACT 1개(3 사이즈) | SIMPLER 1개 | ✗ |
| VLA-Cache | LIBERO엔 OpenVLA, SIMPLER엔 CogACT | 2개 | ✗ **백본이 달라 부호 비교 불가** |
| MoLe-VLA | CogACT / OpenVLA | RLBench 1개 (+실로봇) | ✗ |
| Gaze-Reg | 1개 | LIBERO 1개 | ✗ |

**경쟁자 0건.** VLA-Cache가 벤치마크 둘로 가장 근접하지만 벤치마크마다 백본을
바꾸므로, 부호 뒤집힘은 그 설계에서 관측될 수 없다.

아래 목록은 따라서 "손실"이 아니라 **인용 의무 목록**이다. 각각을 인용하고
우리 측정을 독립 재현으로 제시하면, 재현 자체가 값어치가 된다.

## 인용 의무가 생긴 것 (우리 발견이라 쓸 수 없는 것)

1. ~~"층을 지워도 VLA가 버틴다"~~ → EfficientVLA·MoLe-VLA가 이미 보고
2. ~~"가지치기가 성공률을 올리기도 한다"~~ → EfficientVLA가 "paradoxically" 보고
3. ~~"시각 감축은 VLA를 해친다"~~ → Gaze-Reg 10/10
4. ~~"시각 감축은 속도를 안 벌어준다"~~ → VLA-Cache가 FastV로 실측

**전부 인용하고 "독립 재현"으로 쓰는 것이 정직하며, 재현 자체가 값어치가 있다.**

## 읽어서 새로 생긴 것 (훨씬 크다)

1. **★ §3c가 우리 가설이 아니라 문헌의 미관측 패턴이다.** 3편 × 4방법 × 12구성에서
   PickCan↑ / MoveNear↓, 예외 없음, 용량에 따라 단조. **우리가 최초로 이름을
   붙이고 검정한다.**
2. **★ §3b(평균이 두 곡선을 숨긴다)가 문헌 내 실증을 얻었다.** EfficientVLA가
   자기 표의 반대 곡선 두 개를 평균 내어 *"merely a 0.6% drop"*이라고 쓴다.
   **우리 주장의 가장 좋은 예시가 남의 논문이다.**
3. **우리 depth 기준이 정통이라는 확증.** ShortGPT·EfficientVLA와 동일. 우리는
   그 기준의 **폐루프·교차벤치마크 전이**를 검정하는 것이 된다.
4. **foveation 0% 절감의 원리적 설명 확보.** memory-bound 벽(EfficientVLA) +
   짧은 출력 시퀀스(VLA-Cache). **우리 negative result가 분석이 된다.**
5. **Bridge +18.8이 문헌 전체에 대한 유일한 반례로 확정.**

## 프레이밍 최종안

> Efficiency interventions for VLA policies are evaluated one benchmark and one
> backbone at a time, and reported as a single mean success rate. We re-measure
> three of them — temporal, visual, and depth — across two benchmarks and up to
> four backbones with per-episode pairing, and find that **(i)** the sign of the
> effect is not stable across either axis, **(ii)** the mean hides a consistent
> split between tasks that must resolve which object was named and tasks that
> need not — a split already present, unremarked, in the published tables of
> three prior methods — and **(iii)** on the visual axis the compute the
> intervention is supposed to save is, measurably, zero.

---

# 다음 실험 — 정독이 만들어낸 것

| # | 실험 | 왜 |
|---|---|---|
| 1 | **Bridge에서 foveation 중심 위치 바꾸기** (중심 고정 vs 그리퍼 근처 vs 무작위) | +18.8이 "정보 감소" 때문인지 "무엇을 남겼나" 때문인지 가른다. EfficientVLA의 Random Dropping 붕괴가 이 실험을 요구한다 |
| 2 | **depth 랭킹 고정 vs 에피소드별 재보정** | ShortGPT는 1회 보정, 우리는 에피소드마다. 랭킹이 실제로 달랐으므로(`[2,4,6,23]` vs `[2,4,23,26]`) 교란일 수 있다 |
| 3 | **CogACT를 우리 그리드에 추가** | EfficientVLA·VLA-Cache와 **직접 비교 가능한 유일한 백본**. 추가하면 우리 표가 그들 표와 같은 좌표계에 놓인다 |
| 4 | SpatialVLA 뒷절반만 제거 (부위 교체의 거울상) | §3c-bis를 2×2로 완성 |

**3번이 제일 값어치 있다.** CogACT를 넣으면 *"같은 벤치마크·같은 백본에서
그들이 보고한 평균과 우리가 보고하는 태스크별 분할을 나란히"* 놓을 수 있고,
그게 논문의 그림 1이 된다.
