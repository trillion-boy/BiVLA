# Related Work — 후보군, 그리고 각 논문에 던질 질문

> **⚠️ 이 문서의 모든 서지 정보는 웹 검색 결과에서 왔고, 논문 본문을 직접 읽고
> 확인한 것이 아니다.** 존재·제목·arXiv ID·핵심 주장은 검색 요약 수준에서
> 맞지만, **인용 전에 반드시 원문을 열어 확인해야 한다.** 각 항목의 `[읽음]`
> 칸이 비어 있으면 아직 아무도 그 논문을 읽지 않았다는 뜻이다.
>
> 표기: **[핵심]** = 반드시 읽고 6문항을 채울 것 · **[인용]** = 한 줄 인용이면
> 충분 · **[확인필요]** = 우리 주장과 직접 부딪히므로 원문 대조가 필수

---

## 0. 이 문서가 하는 일

`NeurIPS26_360_Segmentaion.pdf`의 방식을 따른다. 그 문서는 Related Work를
바로 쓰지 않고, **논문 하나당 6개의 고정 질문**에 답한 뒤 그 요약에서 문장을
뽑아냈다. 여기서도 같다 — §3의 6문항 양식을 논문마다 채우고, 채워진 것들에서
§2의 각 문단을 조립한다.

Related Work의 *형식*은 `Bag of Tricks for Inference-time Computation of LLM
Reasoning`(NeurIPS'25 D&B)을 따른다. 그 논문의 Related Work는 1페이지, 3개
테마, 각 테마가 **굵은 리드 문장 → 번호 매긴 접근 나열 → "This work focuses
on…" 한 문장으로 자기 위치 선언**의 형태다.

---

## 1. Related Work의 조직 원리

논문을 나열하는 것이 목적이 아니다. **이 논문이 왜 존재해야 하는지**를
Related Work가 증명해야 한다. 우리의 경우 그 문장은:

> **우리가 쓴 세 개입은 각각 다른 문헌에서 이미 검증된 것을 가져온 것이다.
> 그런데 그 세 문헌의 공통점은, 검증이 전부 열린 루프의 단일 지표 위에서
> 이뤄졌다는 것이다 — Atari 점수, VQA 정확도, perplexity/MMLU. 정책 자신의
> 오차가 다음 관측으로 되먹임되는 닫힌 루프에서, 그리고 벤치마크를 바꿨을 때,
> 이 개입들이 무엇을 하는지는 측정된 적이 없다.**

그래서 각 테마의 마지막 문장은 항상 같은 모양이다 —
*"…but this was validated on X, which does not include Y."* 그 Y가 우리 자리다.

---

## 2. 테마별 후보군

### 2.1 무대 — efficient VLA inference

우리가 왜 이 문제를 다루는지의 배경. 우리 측정으로 OpenVLA는 환경 스텝당
**515 ms**, SpatialVLA는 **937 ms** — 즉 1~2 Hz다. 학습을 다시 하지 않고 이걸
줄이는 것이 실용적 관심사가 된 이유를 이 절이 세운다.

| # | 논문 | 출처 | 역할 | 우선 | 읽음 |
|---|---|---|---|---|---|
| 1 | **A Survey on Efficient Vision-Language-Action Models** | arXiv 2510.24795 | 이 분야의 taxonomy를 그대로 빌려온다. 우리 3축이 이 survey의 어느 칸에 들어가는지 표로 보이면 Related Work가 단번에 정당해진다 | **[핵심]** | |
| 2 | **EfficientVLA: Training-Free Acceleration and Compression for VLA** | OpenReview `SELYlDHZk2` | **언어 모듈의 층 가지치기 + 시각 경로 최적화를 동시에** 한다. 우리 depth 축과 visual 축을 한 논문 안에서 합친 형태라, 가장 직접적인 선행연구 | **[핵심][확인필요]** | |
| 3 | Vision-Language-Action Models: Concepts, Progress, Applications and Challenges | arXiv 2505.04769 | 80편 이상을 정리한 개괄. intro/related의 첫 문장용 | [인용] | |
| 4 | An Anatomy of Vision-Language-Action Models: From Modules to Milestones | arXiv 2512.11362 | 위와 같은 용도. 둘 중 하나만 써도 됨 | [인용] | |
| 5 | **OpenVLA** / **SpatialVLA** / **UniVLA** / **RoboVLMs** 원본 | 각자 | 우리가 실제로 돌린 백본. Setup 절에서 인용하되 Related Work에도 한 줄 | **[핵심]** | |

---

### 2.2a 시간 축 — action repeat / action chunking

**우리가 이걸 고른 이유:** 우리가 발명한 트릭이 아니라 **이 분야가 이미
기본값으로 채택한 유일한 처리량 레버**다. 최신 VLA가 실제로 쓰고 있으므로,
이걸 검정하는 것은 **분야의 전제를 검정하는 것**이다.

**⚠️ 반드시 구분해서 쓸 것:** action **repeat**(1개 예측 → k스텝 유지) ≠
action **chunking**(k개 예측 → k스텝 실행). 우리는 repeat이고, 이는 chunking의
**퇴화 형태(H=1, s=k)**다. 이를 밝히지 않으면 리뷰어가 즉시 짚는다. 오히려
*"우리는 chunking이 줄 수 있는 이득의 하한선을 잰다"*로 쓰면 깔끔하다.

**기원 (연도 무관, 반드시 인용)**

| # | 논문 | 출처 | 역할 | 우선 | 읽음 |
|---|---|---|---|---|---|
| 6 | **DQN (Mnih et al.)** — frame skip = 4가 표준이 된 출발점 | Nature 2015 | 개입의 기원. "행동을 k스텝 유지"라는 아이디어의 원점 | **[핵심]** | |
| 7 | **Frame Skip Is a Powerful Parameter for Learning to Play Atari** (Braylan et al.) | AAAI workshop 2015 | **frame skip 값이 성능을 크게 좌우한다**는 최초의 체계적 관측. 우리 주장(하이퍼파라미터 k의 효과가 불안정)의 직계 조상 | **[핵심]** | |
| 8 | **Dynamic Frame skip Deep Q Network** (FiGAR 계열) | arXiv 1605.05365 | 반복 횟수를 고정하지 않고 학습. 우리가 *고정* k를 쓴 이유를 대비로 설명할 때 필요 | [인용] | |
| 9 | An Analysis of Frame-skipping in Reinforcement Learning | arXiv 2102.03718 | frame skip이 왜 도움이 되는지의 이론적 정리(유효 지평 단축 등) | [인용] | |
| 10 | **ACT (Action Chunking with Transformers)** / **Diffusion Policy** | 2023 | 로봇 모방학습에서 action chunking을 표준으로 만든 두 논문 | **[핵심]** | |

**최신 (2025–2026)**

| # | 논문 | 출처 | 역할 | 우선 | 읽음 |
|---|---|---|---|---|---|
| 11 | **OpenVLA-OFT** — parallel decoding + action chunking | 2025 | VLA에서 chunking이 처리량 표준이 된 지점. 우리가 쓴 OpenVLA의 직계 후속 | **[핵심]** | |
| 12 | **Mixture of Horizons in Action Chunking** | arXiv 2511.19433 | **긴 지평(예측력) vs 짧은 지평(정밀도)의 트레이드오프**를 체계적으로 보임. 우리 repeat 2 vs 4 결과와 정면으로 대화 | **[핵심][확인필요]** | |
| 13 | **DREAM-Chunk: Reactive Action Chunking with Latent World Model** | arXiv 2606.18589 | chunk 실행 중 낡은 관측 문제를 world model로 보정 | [인용] | |
| 14 | **Open-Loop Planning, Closed-Loop Verification: Speculative Verification for VLA** | arXiv 2604.02965 | 같은 문제의 다른 해법 | [인용] | |
| 15 | **VLA-Corrector: Lightweight Detect-and-Correct for Adaptive Action Horizon** | arXiv 2607.01804 | 지평을 적응적으로 조절 | [인용] | |
| 16 | StreamingVLA: Action Flow Matching and Adaptive Early Observation | arXiv 2603.28565 | 스트리밍 실행 | [인용] | |

**이 테마를 닫는 문장 (초안)**

> Action chunking executes on stale observations and does not incorporate
> feedback during the chunk, which is known to accumulate error [13, 14, 15].
> The horizon is therefore a trade-off, and recent work tunes it [12]. **What
> has not been asked is whether the sign of that trade-off is stable when the
> backbone or the benchmark changes** — every such study reports one benchmark
> and one policy family.

---

### 2.2b 시각 축 — foveation (log-polar, blur)

**우리가 이걸 고른 이유:** 로봇은 **행동하는 곳**(그리퍼·대상 주변)에만
고해상도가 필요하고 주변부는 맥락이면 충분하다는 가설. 픽셀 예산을 필요한
곳에 몰아준다.

**변형이 둘인 이유 (설계 논점 — 그대로 논문에 쓸 것):** log-polar는
`cv2.warpPolar`로 **픽셀을 실제로 이동**시키므로 *정보량 감소*와 *기하 왜곡*이
섞인다. blur는 공간 가변 가우시안이라 **어떤 픽셀도 이동하지 않는다.** 따라서
**blur가 "정보량만" 분리해내는 대조군**이다. (근거는 `adaptive_sparse_vla/
foveation.py`의 구현 주석: *"NO pixel is displaced, so intrinsics-based …
geometry intact — unlike log-polar's black frame"*)

**기원 (연도 무관)**

| # | 논문 | 출처 | 역할 | 우선 | 읽음 |
|---|---|---|---|---|---|
| 17 | **Schwartz, complex logarithmic mapping of V1** — `w = log(z + a)` | 1977 / 1980 | **log-polar의 원점.** 망막-피질 사상이 로그 등각사상으로 잘 기술된다는 것. 우리 log-polar 변형의 생물학적 근거 그 자체 | **[핵심]** | |
| 18 | **A review of log-polar imaging for visual perception in robotics** (Traver & Bernardino) | Robotics and Autonomous Systems, 2010 (`S0921889009001687`) | 로봇 비전에서 log-polar가 왜 쓰였는지의 표준 리뷰. "데이터량을 줄이면서 중심 해상도를 지킨다"는 우리 동기의 출처 | **[핵심]** | |
| 19 | Recurrent Models of Visual Attention (glimpse network) | NeurIPS 2014 | 시선 이동을 학습으로 다룬 계보. 우리가 *고정* 중심을 쓴 이유를 대비로 설명 | [인용] | |

**최신 (2025–2026) — 여기가 제일 중요**

| # | 논문 | 출처 | 역할 | 우선 | 읽음 |
|---|---|---|---|---|---|
| 20 | **Gaze-Regularized Vision-Language-Action Models for Robotic Manipulation** | arXiv 2603.23202 | **우리와 가장 가까운 이웃.** 시선 중심으로 foveated RGB를 만들고(중심 고해상도, 주변 다운샘플/블러) VLA에 먹인다. LIBERO-Spatial에서 78.5%. **우리 blur 변형과 사실상 같은 조작** | **[핵심][확인필요]** | |
| 21 | **Look, Focus, Act: Efficient and Robust Robot Learning via Human Gaze and Foveated Vision Transformers** | 2025 | foveated ViT + 인간 시선. 위와 한 쌍 | **[핵심]** | |
| 22 | Eye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loop | 2025 | 시선을 행동으로 학습 | [인용] | |
| 23 | Gaze2Act: Gaze-Conditioned VLA Policies | arXiv 2605.30282 | 시선을 프롬프트로 | [인용] | |
| 24 | Policy-based Foveated Imaging and Perception | arXiv 2606.02565 | foveation 자체를 정책으로 | [인용] | |
| 25 | **FOVI: A biologically-inspired foveated interface for deep vision models** | arXiv 2602.03766 | 생물학적 foveation을 일반 비전 모델 인터페이스로 | [인용] | |
| 26 | Double-Helix Vision (DH-V2): Geometry-Based Visual Sampler for Bandwidth-Constrained Perception | arXiv 2606.14773 | 대역폭 제약하 기하 기반 샘플링 | [인용] | |

**★ 토큰 감축 문헌 — 우리 비용 결과가 여기서 설명된다**

우리 측정에서 foveation은 연산을 **0%** 줄였다(−1.7%, +0.1%, −0.8%, −3.1%,
+0.0%). 이건 실패가 아니라 **이 문헌이 예측하는 결과**다. 아래 논문들이
아끼는 것은 **토큰 수**이고, 우리 foveation은 **픽셀**을 줄이면서 언어 모델에
들어가는 토큰 예산은 그대로 두기 때문이다. Related Work에서 **픽셀 공간 vs
토큰 공간**을 구분해두면 우리 negative result가 *분석*이 되고, 후속 실험(토큰
수준 foveation)이 자연스럽게 도출된다.

| # | 논문 | 출처 | 역할 | 우선 | 읽음 |
|---|---|---|---|---|---|
| 27 | **ToMe (Token Merging)** | ICLR 2023 | 우리 레포에 `tome_siglip.py`로 이미 구현돼 있음 — 실제로 붙어본 문헌 | **[핵심]** | |
| 28 | **FastV** — early-layer attention으로 시각 토큰 가지치기 | ECCV 2024 | 우리 레포에 `fastv_emu3.py` 있음 | **[핵심]** | |
| 29 | SparseVLM — text→vision cross-attention을 중요도 신호로 | 2024/25 | FastV의 대안 | [인용] | |
| 30 | SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference | arXiv 2510.17777 | 최신 VLM 토큰 희소화 | [인용] | |
| 31 | **VLA-Cache: Efficient VLA Manipulation via Adaptive Token Caching** | arXiv 2502.02175 | VLA에 특화. **FastV/SparseVLM이 VLA에서 더 크게 깨진다**고 보고 — 우리 "전이되지 않는다" 주장의 선행 증거 | **[핵심][확인필요]** | |
| 32 | **VLA-Pruner / Bridging the Semantic-Action Gap in Visual Token Pruning** | arXiv 2511.16449 | **"prefill 단계에서 덜 두드러지지만 action decoding에 결정적인 토큰을 성급히 제거한다"** — 우리 §3c(지시 대상 해석 능력이 먼저 죽는다)와 **놀랍도록 가까운 관찰.** 반드시 대조할 것 | **[핵심][확인필요]** | |
| 33 | SpecPrune-VLA: Action-Aware Self-Speculative Pruning | arXiv 2509.05614 | | [인용] | |
| 34 | Token Expand-Merge: Training-Free Token Compression for VLA | arXiv 2512.09927 | | [인용] | |
| 35 | See What Matters: Differentiable Grid Sample Pruning for Generalizable VLA | arXiv 2605.11817 | **grid sample pruning** — 픽셀/샘플 공간에서 자르는 최신 사례. 우리 foveation과 같은 층위 | **[핵심]** | |
| 36 | SAFE-Pruner / VLA-IAP / AVA-VLA | 2605.29662 / 2603.22991 / 2511.18960 | 같은 계열 최신 | [인용] | |

---

### 2.2c 연산 축 — fixed depth pruning

**우리가 이걸 고른 이유 + 반드시 명시할 것:** 우리 랭킹 기준은
`SpatialVLA/experiments/tome/depth_prune_gemma2.py`가 계산하는
**`cos(layer_input, layer_output)`** 이고, *"높은 cos = 잉여"* 로 정렬한다.
이는 **ShortGPT의 Block Influence와 같은 양**이다(BI = 1 − cos(in, out)). 즉
우리가 임의로 만든 휴리스틱이 아니라 **기존 기준을 그대로 채택한 것**이고,
논문에 *"we adopt the Block Influence criterion of ShortGPT"*라고 써야 한다.
이 한 문장이 방법론의 신뢰를 크게 바꾼다.

**왜 adaptive가 아니라 fixed인가:** early-exit류는 스텝마다 판정 비용이 붙고
배치가 깨진다. fixed는 한 번 보정하면 이후 비용이 0이고, **절감량이 지운 층
비율로 예측 가능**하다 — 그리고 우리 비용 측정이 정확히 그것을 확인했다
(32층 중 4층 = 12.5% → **−11.9%**; 8층 = 25% → **−22.6%**; 26층 중 4층 =
15.4% → **−15.9%**). 이건 **우리 데이터로 뒷받침되는 설계 근거**라 강하다.

| # | 논문 | 출처 | 역할 | 우선 | 읽음 |
|---|---|---|---|---|---|
| 37 | **ShortGPT: Layers in LLMs are More Redundant Than You Expect** (Men et al.) | 2024 (OpenReview `JMNht3SmcG`) | **우리 랭킹 기준의 출처.** BI = 1 − cos(in, out). 반드시 원문에서 정의를 대조할 것 | **[핵심][확인필요]** | |
| 38 | **The Unreasonable Ineffectiveness of the Deeper Layers** (Gromov et al.) | 2024 | 깊은 층을 지워도 잘 버틴다는 대표 결과. 우리 OpenVLA "뒷절반만 지우면 오히려 +15.6"과 정확히 일치 | **[핵심]** | |
| 39 | **LayerDrop** (Fan et al.) | ICLR 2020 | 층 단위 제거의 초기 형태 | [인용] | |
| 40 | CALM (Confident Adaptive Language Modeling) / LayerSkip / Depth-Adaptive Transformer | 2020–2024 | **적응형** 깊이. 우리가 fixed를 고른 이유의 대비항 | [인용] | |
| 41 | SkipGPT: Dynamic Layer Pruning with Token Awareness | arXiv 2506.04179 | 최신 동적 층 가지치기 | [인용] | |
| 42 | **Rethinking Layer Redundancy in LLMs: Calibration Objectives and Search for Depth Pruning** | arXiv 2604.24938 | **"calibration이 search보다 중요하다"** — 우리 `--depth-min-layer` 교란(어느 구간을 후보로 두느냐가 결과를 뒤집음)과 같은 종류의 발견 | **[핵심][확인필요]** | |
| 43 | Locality-Aware Redundancy Pruning for LLM Depth Compression | arXiv 2605.27786 | 최신. **어느 구간이냐**가 중요하다는 우리 결과와 대화 | **[핵심]** | |
| 44 | Prune&Comp / Ghosted Layers | 2507.18212 / 2605.15491 | 층 제거 후 보정 | [인용] | |

**VLA에 적용된 층 스킵 (2025–2026)**

| # | 논문 | 출처 | 역할 | 우선 | 읽음 |
|---|---|---|---|---|---|
| 45 | **MoLe-VLA: Dynamic Layer-skipping VLA via Mixture-of-Layers** | arXiv 2503.20384, **AAAI 2026** | **VLA에서 층 스킵을 한 대표 논문.** 10개 태스크 평균 성공률 +8%, 연산 최대 5.6배 절감. 우리 depth 축의 직접 이웃이자 **경쟁 상대** | **[핵심][확인필요]** | |
| 46 | DySL-VLA: Dynamic-Static Layer-Skipping for VLA | arXiv 2602.22896 | 위의 최신 후속 | **[핵심]** | |
| 47 | **Beyond Attention Magnitude: Inter-layer Rank Consistency for Efficient VLA** | arXiv 2603.24941 | 층 중요도 기준을 다시 봄 | [인용] | |
| 48 | Shallow-π: Knowledge Distillation for Flow-based VLAs | arXiv 2601.20262 | 깊이를 줄이는 다른 경로 | [인용] | |

**이 테마를 닫는 문장 (초안)**

> Layer redundancy in language models is well established [37, 38, 39], and the
> Block Influence criterion we adopt comes directly from [37]. Recent work
> brings layer skipping into VLA policies [45, 46]. **But every one of these is
> validated on perplexity, MMLU, or a single manipulation suite — a single
> forward pass, or a single benchmark.** None asks what removing a layer does
> to a policy whose own errors feed back into its next observation for eighty
> steps, and none asks whether the answer survives a change of benchmark.

---

### 2.3 평가 방법론 — 우리 두 번째 기여가 사는 곳

| # | 논문 | 출처 | 역할 | 우선 | 읽음 |
|---|---|---|---|---|---|
| 49 | **SimplerEnv: Evaluating Real-World Robot Manipulation Policies in Simulation** | CoRL 2024 (PMLR v270) | 우리 실험대. **정책 순위가 실제 로봇 순위와 일치함을 검증**했다는 점이 우리 결과의 무게를 결정하므로, 그 검증 범위를 정확히 인용해야 함 | **[핵심][확인필요]** | |
| 50 | **PhAIL: A Real-Robot VLA Benchmark and Distributional Methodology** | arXiv 2605.29710 | **★ 우리 평가 기여를 정당화하는 결정적 인용.** 실제 로봇 VLA 평가가 *"고정 timeout에서의 이진 성공률, 조건당 N < 25, 신뢰구간이나 paired 검정은 거의 없음"*이라고 명시. **우리가 채우는 공백을 남이 진술해준 것** | **[핵심][확인필요]** | |
| 51 | RobotArena ∞: Scalable Robot Benchmarking via Real-to-Sim Translation | arXiv 2510.23571 | 평가 규모 확장 | [인용] | |
| 52 | REALM: A Real-to-Sim Validated Benchmark for Generalization | arXiv 2512.19562 | 일반화 평가 | [인용] | |
| 53 | Experiences from Benchmarking VLA Models for Robotic Manipulation | arXiv 2511.11298 | 벤치마킹 실무 경험담 | [인용] | |
| 54 | **Encoder Winners Do Not Reliably Transfer Across VLA Backbone Scale** | arXiv 2606.14153 | **★ 우리와 가장 가까운 주장.** 작은 백본에서 이긴 인코더가 큰 백본에서 이기지 않는다. **우리는 "인코더 선택 × 백본 규모"가 아니라 "추론 시점 개입 × 벤치마크·백본"이라는 차별점을 명시해야 함** | **[핵심][확인필요]** | |
| 55 | VLM4VLA: Revisiting Vision-Language-Models in VLA Models | arXiv 2601.03309 | VLM 선택이 VLA 성능으로 어떻게 전이되는지 | [인용] | |
| 56 | **Deep Reinforcement Learning that Matters** (Henderson et al.) | AAAI 2018 | 재현성 위기의 고전. "시드 몇 개" 관행의 출처 | **[핵심]** | |
| 57 | **Deep RL at the Edge of the Statistical Precipice / rliable** (Agarwal et al.) | NeurIPS 2021 | 소표본 RL 평가의 표준 처방(층화 부트스트랩, IQM). **우리는 이것의 대안이 아니라 다른 조건(결정론·paired)에 맞는 도구를 쓴다**고 위치를 잡아야 함 | **[핵심]** | |
| 58 | VLAConf: Calibrated Task-Success Confidence for VLA | arXiv 2605.29605 | 성공 확률 보정 | [인용] | |

**이 테마를 닫는 문장 (초안)**

> Evaluation practice for VLA policies is known to be statistically thin: [50]
> surveys real-robot papers and finds binary success at a fixed timeout with
> N < 25 per condition, almost never with confidence intervals or a paired
> comparison. The RL literature's answer to small samples is stratified
> bootstrap and interval estimates [57], designed for **stochastic training**
> where seed variance dominates. **Our setting is different in a way that
> admits a stronger answer**: greedy decoding into a seeded simulator is
> deterministic — we verify 85/85 episodes reproduce exactly, including step
> counts — so run-to-run variance is zero and the only remaining uncertainty is
> which episodes the protocol drew. That is precisely what a paired McNemar
> exact test on discordant pairs measures, which makes the reported p-value the
> **complete** account of uncertainty rather than a partial one.

---

### 2.4 장르 — empirical study / bag of tricks

| # | 논문 | 출처 | 역할 | 우선 | 읽음 |
|---|---|---|---|---|---|
| 59 | **Bag of Tricks for Inference-time Computation of LLM Reasoning** (Liu et al.) | NeurIPS 2025 D&B | **우리 논문의 형식 모델.** 특히 *"improvements are not always additive when different techniques are combined"* — 우리 `prune2 + repeat2` 결과와 **같은 발견을 다른 도메인에서** 한 것. 반드시 그렇게 인용 | **[핵심]** | |
| 60 | **Bag of Tricks for Image Classification with CNNs** (He et al.) | CVPR 2019 | 이 장르의 원조 | [인용] | |

---

## 3. 논문마다 채울 6문항 (360 문서 양식)

각 **[핵심]** 논문마다 아래를 채운다. 채우고 나면 Related Work 문장은
2·3·6번에서 거의 그대로 나온다.

```
### <제목> [<venue> <year>] <url>

0. 우리 논문에서 이 인용이 맡는 역할  ← 360 문서에는 없지만 우리에겐 필수
   (예: "우리 depth 랭킹 기준의 출처" / "우리가 반박할 검증 조건")

1. 어떤 연구 문제를 다루는가?

2. 그 문제를 어떤 방식으로 제기했는가? 새로운 문제인가 방향인가?

3. 핵심 아이디어는 무엇이고, 어떤 가정 위에 서 있는가?
   ← 이 "가정"이 제일 중요하다. 우리가 깨는 것이 대개 여기 있다.

4. 기술적 기여는 무엇이고, 각 모듈이 어떤 난점을 푸는가?

5. 효과를 검증하기 위해 어떤 비교 실험을 했는가?
   ← 벤치마크 개수, 백본 개수, 조건당 N, 통계 검정 유무를 반드시 적을 것.
     우리 Related Work의 마지막 문장이 전부 여기서 나온다.

6. ablation은 어떻게 했는가? 주장을 뒷받침하기에 충분한가?

7. 우리와의 차이 한 문장  ← 추가 항목
```

**5번을 표로도 모아둘 것.** 이 표가 곧 Related Work의 근거이자, Introduction의
"기존 연구는 X만 봤다"의 증거가 된다:

| 논문 | 벤치마크 수 | 백본 수 | 조건당 N | 통계 검정 | paired? |
|---|---|---|---|---|---|
| (채울 것) | | | | | |
| **본 연구** | **2** | **3–4** | **96 / 135** | **McNemar exact + Fisher + Bonferroni** | **예** |

---

## 4. 각 method의 "왜 이걸 골랐나" — 논문에 들어갈 초안 문장

멘토가 요구한 부분이다. 각 3~4문장, Related Work 각 소절의 첫 문단이 된다.

### Temporal — action repeat

> Holding one action for several environment steps is the oldest efficiency
> lever in this literature: it is the frame-skip of DQN [6], and Braylan et al.
> [7] showed early that the skip length is a decisive parameter rather than an
> implementation detail. In manipulation the same idea reappears as action
> chunking, where a policy predicts a short horizon and executes it open-loop
> [10], and it is now the standard throughput mechanism in VLA systems [11].
> **We adopt its degenerate form — predict one action, hold it for k steps —
> deliberately**, because it isolates the temporal variable: any change in
> success is attributable to acting on a stale observation, not to a different
> action-prediction architecture. It also gives the lower bound on what
> chunking can buy, since a chunk of k predicted actions cannot be worse
> informed than one action repeated k times.

### Visual — foveation

> Biological vision does not sample the field uniformly, and the retinotopic
> map is well described by a complex logarithm [17] — the observation that
> motivated log-polar sensing in robotics as a way to shrink data volume while
> preserving foveal detail [18]. The same intuition applies to manipulation:
> **acuity is only needed where the gripper acts, and the periphery only has to
> supply context.** Recent VLA work makes this concrete by building foveated
> observations around a gaze estimate [20, 21]. **We use two variants rather
> than one, because they separate two things that log-polar sampling
> confounds.** Log-polar removes information *and* displaces every pixel;
> space-variant blur removes information while leaving geometry intact. The
> blur variant is therefore the control that isolates information loss from
> geometric distortion.
>
> A separate line reduces visual cost at the *token* level rather than the
> pixel level [27, 28, 31, 32]. **The distinction matters for cost and we
> return to it in §<cost>:** our foveation reduces pixels but leaves the token
> budget entering the language model unchanged, and we measure that it saves
> essentially no compute — which is what this literature predicts.

### Depth — fixed layer pruning

> Transformer layers are individually far less important than their count
> suggests [37, 38, 39], and ShortGPT [37] makes this operational with Block
> Influence, one minus the cosine similarity between a layer's input and output
> hidden states. **We adopt exactly this criterion**, calibrating the ranking
> once on the first step of an episode and thereafter bypassing the N
> highest-similarity layers at zero additional cost. **We use fixed rather than
> adaptive depth [40, 41, 45] on purpose**: an adaptive router adds a per-step
> decision whose own cost and variance would confound the measurement, whereas
> a fixed bypass has a compute saving that is predictable from the layer
> fraction alone — which our timing confirms (four of 32 layers is 12.5% of the
> stack and costs 11.9%; eight is 25% and costs 22.6%).
>
> **⚠️ 여기에 우리 §3c-bis 발견을 한 문장 넣을 것:** 어느 *구간*을 후보로
> 두느냐가 같은 이름의 실험을 다른 실험으로 만든다는 것. [42, 43]과 직접
> 연결된다.

---

## 5. 작업 순서

1. **[핵심] 표시된 것 먼저 읽고 §3의 6문항을 채운다.** 특히 `[확인필요]`가
   붙은 8편(2, 12, 20, 31, 32, 37, 42, 45, 49, 50, 54)은 우리 주장과 직접
   부딪히므로 원문 대조가 필수다.
2. §3 하단의 **"벤치마크 수 / 백본 수 / N / 검정" 표**를 채운다. 이 표 하나가
   Introduction과 Related Work의 논거를 동시에 만든다.
3. §4의 초안 문장을 읽은 내용에 맞게 고친다.
4. Related Work 최종본을 쓴다 (목표: 1~1.25 페이지).
5. 그 다음에 종합 보고서 §3–§6.

**지금 가장 위험한 것:** #54(Encoder Winners Do Not Reliably Transfer)와
#32(VLA-Pruner의 "action decoding에 결정적인 토큰을 성급히 제거한다")가 우리
주장과 얼마나 겹치는지. 겹침이 크면 우리 novelty 진술을 좁혀야 한다. **이 두
편을 제일 먼저 읽을 것.**
