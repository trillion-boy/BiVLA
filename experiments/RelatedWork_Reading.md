# 선행연구 정독 결과 — 우리 세 개입은 VLA에 실제로 도움이 되는가

> **⚠️ 접근 제한을 먼저 밝힌다.** 이 세션의 egress 정책이 `arxiv.org`,
> `openreview.net`, `semanticscholar.org`, `alphaxiv.org`, `huggingface.co`를
> 전부 차단한다. **PDF 원문을 열지 못했다.** 아래 내용은 전부 **검색 엔진이
> 반환한 논문 요약과 인용 스니펫**에서 나왔다. 수치는 그 스니펫이 명시한
> 값이고, 문맥(어떤 조건에서 잰 값인지)은 확인하지 못한 경우가 있다.
> **논문에 인용하기 전에 원문 대조가 반드시 필요하다.** 특히 아래 굵은
> 숫자들이 그렇다.

---

## 요약 — 세 줄

1. **시각 축(foveation): 문헌은 "VLA에 해롭다"로 수렴한다.** 독립된 그룹이
   다른 백본·다른 벤치마크에서 **−7.4**를 보고했고, 토큰 가지치기 계열도
   VLA로 옮기면 깨진다고 반복 보고된다. **우리 Fractal 결과(−19.3)는 문헌과
   일치하고, 우리 Bridge 결과(+18.8)는 문헌이 예측하지 못한다.** 후자가 우리
   논문의 진짜 자산이다.
2. **연산 축(depth pruning): 문헌은 "도움이 된다"고 한다 — 단 한 벤치마크에서.**
   MoLe-VLA가 RLBench 10태스크에서 **+9.7% 성공률 + 36.8% 가속**을 보고한다.
   우리 OpenVLA 결과(+15.6, prune 8에서 ±0)는 이와 일치하지만, **SpatialVLA에서
   같은 조작이 −17.8이다.** 문헌은 백본을 하나만 봤다.
3. **시간 축(action repeat): 문헌은 "처리량은 벌지만 오차가 누적된다"고 한다.**
   우리와 일치한다. 우리가 더한 것은 **그 트레이드오프의 부호가 백본마다
   뒤집힌다**는 것이다(SpatialVLA +12.5 vs UniVLA −70.8, 같은 Bridge).

---

## 1. 시각 축 — foveation은 VLA에 도움이 되는가?

### 1.1 [핵심] Gaze-Regularized Vision-Language-Action Models for Robotic Manipulation

**출처:** arXiv 2603.23202 · **CVPR 2026 Workshop (GRAIL-V)**
(`openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/papers/Pani_...`)

**무슨 내용인가 (검색 유래)**
시선 분포를 VLA 학습에 정규화 항으로 넣는다. 그리고 **별도의 변형으로, 시선
분포의 peak를 중심으로 foveated RGB 이미지를 만들어 입력으로 쓴다** — 중심은
고해상도, 주변은 다운샘플/블러. 즉 **우리 blur 변형과 사실상 같은 조작이다.**

**핵심 수치 (검색 유래, 원문 확인 필요)**

| 조건 | LIBERO-Spatial 성공률 |
|---|---|
| baseline | **85.9%** |
| gaze **정규화**(학습 신호로 사용) | **95.5%** (+9.6) |
| **foveated RGB 입력**(주변부 제거) | **78.5%** (**−7.4**) |

**저자들의 해석 (검색 유래):** 다시점 조작 환경에서 주변부 디테일을 공격적으로
줄이면 **정책이 정밀한 공간 추론에 쓰던 맥락 단서**(테이블 기하, 지지면, 대안
grasp)가 사라진다.

**우리와의 연결 — 이게 제일 중요하다**

- **구분해야 할 것:** 시선을 *학습 신호*로 쓰는 것(+9.6)과 주변 픽셀을 *추론
  시점에 제거*하는 것(−7.4)은 완전히 다른 개입이다. **우리 것은 후자다.**
  이 구분을 안 하면 "foveation은 좋다"는 잘못된 인용이 된다.
- **독립 재현:** 다른 백본, 다른 벤치마크(LIBERO), 다른 그룹이
  **foveation이 VLA를 해친다**는 우리 Fractal 결과와 같은 방향을 얻었다.
  크기도 비슷한 수준이다(그들 −7.4 / 우리 blur −8.9, log-polar −19.3).
- **그런데 우리에겐 반대 부호가 있다.** OpenVLA/Bridge에서 log-polar **+18.8**,
  blur **+17.7**. **문헌 어디에도 이 방향의 보고가 없다.**

> **Related Work에 그대로 들어갈 문단 (초안)**
>
> Recent work reports that foveating a VLA's observation is harmful: [Gaze-Reg]
> builds a foveated RGB input around a gaze peak and measures 78.5% against an
> 85.9% baseline on LIBERO-Spatial, attributing the 7.4-point cost to the loss
> of peripheral context that the policy uses for spatial reasoning. We reproduce
> that cost — −8.9 with blur and −19.3 with log-polar on Fractal — **and also
> its opposite: on Bridge the same operation gains 18.8 points.** Neither
> measurement is wrong. They are measurements of different benchmarks, and
> reporting either alone licenses a conclusion the other refutes.

### 1.2 [핵심] VLA-Cache — 그리고 "픽셀을 줄여도 연산은 안 준다"의 선례

**출처:** arXiv 2502.02175 · `vla-cache.github.io`

**핵심 주장 (검색 유래)**
- VLA-Cache는 프레임 간 정적 토큰을 캐시 재사용. **FLOPs −27.31%, 1.63× 속도,
  성공률 하락 0.3%**.
- **FastV와 SparseVLM은 "추론 속도를 개선하지 못하고 태스크 성능을 자주
  떨어뜨린다."**
- **FastV는 attention 계산에서 토큰을 마스킹할 뿐 GPU 작업량을 줄이지 않고,
  오히려 마스킹 오버헤드를 더한다.** SparseVLM은 pruning/merging/recycling에서
  추가 비용이 든다.
- 이들의 전략은 **단일 프레임 내에서 작동하며, 정밀 조작에 결정적인 spatial
  fidelity를 깬다.**
- 정적 토큰을 전부 재사용하면 성공률이 **74.2%**로 떨어진다 — 시각적 유사성만으로는
  로봇 제어에서 재사용 판단이 불충분하다.

**우리와의 연결 — 우리 비용 결과의 직계 선례다**

우리 측정에서 foveation은 스텝당 연산을 **0%** 줄였다(−1.7, +0.1, −0.8, −3.1,
+0.0). 우리는 이걸 *"픽셀은 줄였지만 언어 모델에 들어가는 토큰 예산은 그대로라
아낄 것이 없다"*로 설명했다. **VLA-Cache가 FastV에 대해 같은 종류의 실패를
보고한다** — 마스킹은 이론적 FLOPs를 줄이지만 실제 GPU 작업은 안 줄인다.

> **공통 교훈, 그리고 우리가 더하는 것:** 시각 개입은 **"무엇을 제거했는가"와
> "무엇이 비용인가"가 어긋나기 쉽다.** [VLA-Cache]는 attention 마스킹에서 이를
> 보였고, 우리는 **픽셀 공간**에서 같은 어긋남을 측정한다 — 그리고 우리 쪽이
> 더 극단적이다. 픽셀의 80%를 버려도 스텝당 연산은 통계적으로 0이다.

### 1.3 [핵심] VLA-Pruner / Bridging the Semantic-Action Gap

**출처:** arXiv 2511.16449 (v1 제목 *VLA-Pruner*, v4 제목 *Bridging the
Semantic-Action Gap…*) · 코드 `github.com/MINT-SJTU/VLA-Pruner`

**핵심 주장 (검색 유래)**
- VLM용 토큰 가지치기는 **prefill attention** 같은 semantic salience만으로
  토큰을 고른다. 그런데 VLA는 **고수준 semantic 이해 + 저수준 action 실행**의
  이중 체계다.
- 그래서 기존 방법은 **semantic 단서 쪽으로 토큰 보존이 편향되고, action 생성에
  결정적인 정보를 버린다.**
- prefill attention과 action-decode attention의 패턴이 **뚜렷하게 다르다.**
- 두 신호를 결합해 1.99× 가속, 조작 품질 유지.

**우리와의 연결 — 겹치지 않는다. 오히려 방향이 반대다**

| | 무엇을 제거하나 | 무엇이 살아남나 | 무엇이 죽나 |
|---|---|---|---|
| **VLA-Pruner가 지적하는 기존 방법** | 시각 **토큰**(semantic salience 기준) | semantic 이해 | **action 생성** |
| **우리 §3c** | 디코더 **용량**(층) 또는 픽셀 | **파지(motor)** — 멀쩡하거나 향상 | **지시 대상 해석(referential)** |

**정반대다.** 그들은 *"semantic은 지키고 action을 깬다"*, 우리는 *"motor는
지키고 semantic(어떤 물체인지)을 깬다"*.

이건 novelty 위협이 아니라 **더 강한 주장의 재료**다: **무엇이 먼저 죽는지는
무엇을 제거했는지에 달려 있다.** 토큰을 attention으로 자르면 action이 죽고,
디코더 용량을 자르면 grounding이 죽는다. 두 결과를 나란히 놓으면 *"효율 개입은
단일 스칼라 '성능'을 깎는 게 아니라 특정 능력을 선택적으로 제거한다"*가 된다.

### 1.4 [인용] 그 밖의 수렴 증거

- **일반화 관찰 (검색 유래):** *"visual token pruning 방법들은 VLA로 옮기면
  성능이 불만족스럽다. VLM은 global semantics에 집중하지만 로봇 태스크는 local
  semantics에 더 의존하기 때문이다."*
- LAC(arXiv 2602.00686): 성공률 **75.0 → 76.9 (+1.9)**, 1.76× 속도, FLOPs −25.3%
- SAFE-Pruner(2605.29662): **74.5% vs vanilla CogACT 74.8%** (−0.3), FLOPs 37.4%, 1.73×
- SpecPrune-VLA(2509.05614): 1.46–1.57× 속도, "성공률 저하 미미"

**주의해서 읽을 것:** 위 세 편은 전부 **LIBERO 한 벤치마크**이고, 보고 형식이
*"성공률은 거의 유지하면서 속도를 벌었다"*이다. **신뢰구간도 paired 검정도
없다.** 이게 정확히 PhAIL이 지적하는 관행이고, 우리 §5.1(부호 불안정)이 왜
필요한지의 근거다.

### 1.5 시각 축 결론 — 우리 질문에 대한 답

> **foveation은 VLA에 도움이 되는가?**
> **문헌의 답: 아니다.** 독립 그룹이 −7.4를 보고했고, 토큰 가지치기 계열도
> VLA에서 반복적으로 깨진다. 그리고 **비용도 안 아낀다** — 우리 측정 0%,
> VLA-Cache가 FastV에 대해 같은 관찰.
>
> **우리의 답: 대개 아니지만, 항상은 아니다.** OpenVLA/Bridge에서 **+18.8**은
> 문헌이 설명하지 못하는 결과다. 이건 이 논문에서 **버리면 안 되는 이상치**다.

**⚠️ 아직 설명 못 한 것.** "baseline이 낮으면 도움이 되는가"를 우리 데이터로
확인해보면 단조롭지 않다:

| 칸 | baseline | log-polar Δ | blur Δ |
|---|---|---|---|
| OpenVLA / Bridge | 15.6% | **+18.8** | **+17.7** |
| OpenVLA / Fractal | 38.5% | **−19.3** | −8.9 |
| SpatialVLA / Fractal | 84.4% | +0.7 | −1.5 |
| *(참고)* Gaze-Reg / LIBERO-Spatial | 85.9% | — | **−7.4** |

15.6%에서 크게 오르고, 38.5%에서 크게 떨어지고, 84.4%에서는 아무 일도 안
일어난다. **단조롭지 않으므로 "약한 정책일수록 도움된다"는 설명은 성립하지
않는다.** 정직하게 열어두고, Discussion에 열린 질문으로 적어야 한다.

---

## 2. 연산 축 — depth pruning은 VLA에 도움이 되는가?

### 2.1 [핵심] MoLe-VLA — 문헌에서 가장 강한 긍정 사례

**출처:** arXiv 2503.20384 · **AAAI 2026** (`ojs.aaai.org/.../38945`) ·
코드 `github.com/RoyZry98/MoLe-VLA-Pytorch`

**핵심 수치 (검색 유래, 원문 확인 필요)**
- **RLBench 10개 시뮬레이션 태스크**에서 평균 성공률 **+9.7%**, OpenVLA 대비
  추론 **36.8% 가속**. (다른 스니펫은 "10개 태스크 평균 +8%, 연산 최대 5.6×
  절감"이라고 함 — **두 수치가 다르므로 원문에서 확정할 것**)
- **층 스킵 내성:** *"상당한 층 스킵에도 대부분의 태스크에서 성능이 안정적이고,
  24층까지는 성공률 저하가 미미하다. 30층을 스킵할 때(FLOPs 약 95% 감소) 비로소
  큰 저하가 온다."*

**우리와의 연결**

- **우리 OpenVLA 결과와 정확히 일치한다.** OpenVLA/Bridge에서 `depth prune 8`
  (32층 중 8층)이 **+0.0** — 아무 일도 안 일어난다. OpenVLA/Fractal에서
  `depth prune 4`가 **+15.6**. 즉 **OpenVLA는 층 제거를 매우 잘 견딘다**는
  MoLe-VLA의 관찰을 독립적으로 재현한 셈이다.
- **그런데 SpatialVLA에서는 반대다.** `depth prune 1`(26층 중 1층!)이 Bridge에서
  **−10.4**, `depth prune 4`가 Fractal에서 **−17.8**. **한 층을 지워도 깨진다.**
- **중요한 차이:** MoLe-VLA는 라우터(STAR)를 **학습**하고 인지 능력 손실을
  distillation(CogKD)으로 보상한다. 즉 **training-free가 아니다.** 우리는
  training-free fixed pruning이다. 우리 결과는 *"학습 없이 층을 지우면 백본에
  따라 결과가 갈린다"*이고, 이는 MoLe-VLA를 반박하지 않고 **그 라우터와
  distillation이 왜 필요한지를 설명해준다.**

> **Related Work 문장 (초안)**
>
> Layer skipping has been brought into VLA policies with strong results:
> [MoLe-VLA] reports +9.7% mean success and 36.8% faster inference on ten
> RLBench tasks, and finds performance stable up to 24 of the decoder's layers
> skipped. **Two properties of that result bound what it establishes.** It uses
> a learned router and a distillation objective to recover the capability that
> skipping removes, so it is not training-free; and it is measured on one
> benchmark and one policy family. **Training-free, fixed-schedule pruning on a
> second backbone behaves differently in our measurements** — removing a single
> layer of SpatialVLA's 26 costs 10.4 points on Bridge — which suggests the
> router and the distillation are not incidental to the reported gain.

### 2.2 [핵심] ShortGPT — 우리 랭킹 기준의 출처

**출처:** Men et al., 2024 · OpenReview `JMNht3SmcG`

**정의 (검색 유래, 여러 출처가 일치)**
> ShortGPT assigns each layer a **Block Influence (BI)** score defined as
> **one minus the cosine similarity between its input and output hidden
> states**, averaged over a calibration set. Layers with low BI (high
> input-output similarity) are considered redundant and removed.

**우리 코드 대조 (`SpatialVLA/experiments/tome/depth_prune_gemma2.py`)**
```
cs = cosine_similarity(layer_input, layer_output).mean()
# higher cos = more redundant = safer to drop; rank descending
```
**동일하다.** 우리는 BI를 그대로 쓰되 `1 −`를 생략하고 내림차순으로 정렬한
것이므로, 랭킹은 ShortGPT와 **같은 순서**다.

**→ 논문에 반드시 이렇게 쓸 것:**
> We rank layers by the Block Influence criterion of [ShortGPT] — the cosine
> similarity between a layer's input and output hidden states — calibrated once
> on the first step of an episode.

이 한 문장이 없으면 "임의 휴리스틱"으로 읽히고, 있으면 "기존 기준의 폐루프
전이 검정"이 된다. **논문의 성격이 바뀐다.**

### 2.3 [핵심] Rethinking Layer Redundancy: Calibration Matters More Than Search

**출처:** arXiv 2604.24938 (v1/v3 제목이 다름)

**핵심 주장 (검색 유래):** 깊이 가지치기에서 **어떤 calibration 목적함수를
쓰느냐가, 어떤 탐색 알고리즘을 쓰느냐보다 중요하다.**

**우리와의 연결 — §3c-bis가 이 문헌의 사례다**

우리는 `--depth-min-layer`가 **후보 구간**을 정하는 것만으로 같은 이름의 실험이
다른 실험이 된다는 것을 측정했다. 층 개수를 4로 고정하고 **부위만** 바꿨더니:

| | 뒷절반만 제거 → 앞쪽 포함 제거 | p |
|---|---|---|
| pick 계열 (n=75) | 41.3% → 40.0% (**−1.3**) | 1.0000 |
| `move_near` (n=60) | 70.0% → 50.0% (**−20.0**) | **0.0169** |

**"calibration/후보 설정이 결과를 지배한다"의 폐루프 판본**이다. 이 논문과
[Locality-Aware Redundancy Pruning](arXiv 2605.27786)을 함께 인용하면 우리
§3c-bis가 고립된 버그 수기가 아니라 **문헌이 이미 지목한 현상의 로봇 사례**가
된다.

### 2.4 연산 축 결론

> **depth pruning은 VLA에 도움이 되는가?**
> **문헌의 답: 그렇다 — OpenVLA 계열에서, 학습된 라우터와 함께, 한 벤치마크에서.**
> **우리의 답: 백본에 따라 갈린다.** OpenVLA는 8층을 지워도 무사하고 4층에서
> +15.6까지 간다. SpatialVLA는 1층에서 −10.4다. **그리고 이건 연산을 실제로
> 아끼는 유일한 "정직한" 축이다** — 지운 층 비율만큼 정확히 준다(12.5% → −11.9%,
> 25% → −22.6%).

---

## 3. 시간 축 — action repeat은 VLA에 도움이 되는가?

### 3.1 문헌의 진단 (검색 유래)

- **핵심 결함:** *"action chunking은 낡은 관측(stale observations)에 기반해
  행동하고 chunk 실행 중 새로 관측된 피드백을 반영하지 못해, 오차가 누적되고
  제어 성능이 저하된다."*
- **트레이드오프의 형태:** [Mixture of Horizons, arXiv 2511.19433]가
  *"장기 예측력(long-term foresight)과 단기 정밀도(short-term precision) 사이의
  트레이드오프"*를 체계적으로 보고하고, 지평을 섞어서 완화한다.
- **처방들:** 반응형 chunk 재계획([DREAM-Chunk] 2606.18589), 투기적 검증
  ([Speculative Verification] 2604.02965), 적응형 지평
  ([VLA-Corrector] 2607.01804), 스트리밍 실행([StreamingVLA] 2603.28565).
  **2026년에 이 문제를 고치려는 논문이 최소 4편 나왔다** — 문제가 실재한다는
  방증이다.

### 3.2 우리와의 연결

- **일치:** 우리 repeat 4가 SpatialVLA/Fractal에서 **−40.0**, OpenVLA/Bridge에서
  **−11.5**. "오차 누적"이 정확히 이 모양이다.
- **우리가 더하는 것:** 문헌은 트레이드오프의 **크기**를 다루고, 우리는 그
  **부호가 백본마다 뒤집힌다**는 것을 보인다. 같은 Bridge, 같은 repeat 2에서
  SpatialVLA **+12.5**, OpenVLA **−8.3**, UniVLA **−70.8**. 셋이 한 화면에 있다.
- **⚠️ 반드시 밝힐 것:** 우리는 action **repeat**(1개 예측 → k스텝 유지)이고
  문헌의 action **chunking**(k개 예측 → k스텝 실행)이 아니다. repeat은
  chunking의 퇴화형(H=1, s=k)이고, **chunking이 줄 수 있는 이득의 하한**이다.
  이 문장을 안 넣으면 리뷰어가 즉시 짚는다.

### 3.3 시간 축 결론

> **action repeat은 VLA에 도움이 되는가?**
> **문헌의 답: 처리량은 확실히 벌지만, 낡은 관측 때문에 제어가 나빠진다.
> 2026년에 이를 고치려는 논문이 여럿이다.**
> **우리의 답: 일치한다. 그리고 우리가 재는 세 축 중 연산을 압도적으로 가장
> 많이 아끼는 축이면서(−50% / −75%), 동시에 캠페인 최악의 실패(−70.8)를
> 만드는 축이다.**

---

## 4. 평가 방법론 — 우리 두 번째 기여의 자리

### 4.1 [핵심] PhAIL: A Real-Robot VLA Benchmark and Distributional Methodology

**출처:** arXiv 2605.29710 (2026)

**핵심 주장 (검색 유래)**
> Real-world evaluation of VLA policies **still relies on binary success rate at
> a fixed timeout with N < 25 rollouts per condition, almost always without
> confidence intervals or paired statistical comparison.**

**그들의 처방:** time-to-success CDF를 기본 단위로 삼고, 점수(Human-Relative
Throughput + 부트스트랩 신뢰구간)와 유의성 검정(Kolmogorov–Smirnov, 객체별
계산 후 macro 평균)을 **분리**한다. N ≤ 30에서 이진 지표가 못 가르는 두 비교를
가른다.

**우리와의 관계 — 경쟁이 아니라 상보다**

| | PhAIL | 본 연구 |
|---|---|---|
| 대상 | 실제 로봇 (확률적) | real-to-sim (결정론적) |
| 짝짓기 | 불가 — 에피소드가 재현되지 않음 | **가능** — 85/85 비트 단위 재현 |
| 검정 | KS (분포 비교, unpaired) | **McNemar exact (paired)** + Fisher |
| 불확실성 | 부트스트랩 CI | **p값이 전부** (재실행 분산 = 0) |

> **Related Work 문장 (초안)**
>
> [PhAIL] surveys real-robot VLA evaluation and finds it rests on binary success
> at a fixed timeout, N < 25 per condition, almost never with a confidence
> interval or a paired comparison; it proposes a distributional remedy for the
> stochastic real-robot setting. **Our setting admits a stronger one.** Greedy
> decoding into a seeded simulator is deterministic — we verify that 85 of 85
> episodes reproduce exactly, including step counts and grasp flags — so
> re-run variance is zero and the only remaining uncertainty is which episodes
> the protocol drew. A paired exact test on discordant pairs measures precisely
> that, which makes the reported p-value the complete account of uncertainty
> rather than a partial one.

### 4.2 [핵심] Encoder Winners Do Not Reliably Transfer Across VLA Backbone Scale

**출처:** arXiv 2606.14153 (2026)

**핵심 주장 (검색 유래)**
- 작은 백본에서 이긴 인코더가 큰 백본의 상위권을 고르지 못한다. SmolVLA-450M
  에서 최저 MSE는 SigLIP, π₀.₅-3.3B의 libero_spatial에서는 DINOv2-small.
- π₀.₅-libero_object에서는 상위 3개가 **근사 동률**이고 **top-1 정체성이 시드
  섭동에 안정적이지 않다.**
- **평가 지표: offline action-MSE**. 이유가 명시돼 있다 — *"공개된 VLA 체크포인트와
  대상 시뮬레이터 사이의 embodiment mismatch 때문에 closed-loop rollout 성공률이
  붕괴하기 때문."*

**novelty 판정: 겹치지 않는다. 그리고 우리에게 유리하다**

| | 그들 | 우리 |
|---|---|---|
| 바꾸는 변수 | **비전 인코더** 선택 | **추론 시점 개입** |
| 가로지르는 축 | 백본 **규모** | **벤치마크** + 백본 **계열** |
| 지표 | offline action-**MSE** | **closed-loop 성공률**, paired |
| closed-loop | **회피함** (붕괴한다고 명시) | **이것이 측정 대상** |

**우리는 그들이 못 한 것을 한다.** 그러니 위협이 아니라 **같은 현상의 독립
사례**로 인용해서 우리 주장을 넓히는 데 쓴다:

> The instability we report is not confined to inference-time interventions.
> [Encoder Winners] finds the same in a different variable — the best vision
> encoder at one backbone scale is not the best at another, and at one suite
> the top-1 identity is not even stable under seed perturbation. **That study
> reports offline action-MSE explicitly because closed-loop success collapses
> under embodiment mismatch; ours is a closed-loop paired measurement, which is
> what makes the sign of the effect observable at all.**

---

## 5. 그래서 우리 논문이 무엇을 주장할 수 있는가

### 5.1 문헌이 이미 아는 것 (= 우리 기여가 아닌 것)

- 시각 감축은 VLA 조작을 해친다 — **이미 보고됨** (−7.4, LIBERO)
- 토큰 가지치기는 이론 FLOPs와 실제 속도가 어긋난다 — **이미 보고됨** (FastV)
- action chunking은 낡은 관측 때문에 오차가 누적된다 — **이미 보고됨**, 2026년
  처방 논문 4편+
- 층 제거는 OpenVLA에서 놀랍도록 잘 버틴다 — **이미 보고됨** (24/32층)
- VLA 평가는 통계적으로 빈약하다 — **이미 보고됨** (PhAIL)

**우리 논문이 위 다섯 개를 "발견"이라고 쓰면 안 된다.** 각각을 인용하고, 우리
데이터가 **독립 재현**임을 밝히는 것이 정직하고 또 유리하다(재현은 그 자체로
가치가 있다).

### 5.2 문헌이 모르는 것 (= 진짜 기여)

1. **같은 개입의 부호가 벤치마크를 바꾸면 뒤집힌다.** foveation +18.8 / −19.3,
   상호작용 p = 0.0000055. **어떤 선행 연구도 두 벤치마크로 재지 않았으므로
   관측될 수 없었다.**
2. **부호가 백본을 바꿔도 뒤집힌다.** 같은 Bridge에서 repeat 2가 SpatialVLA
   +12.5, UniVLA −70.8.
3. **OpenVLA/Bridge foveation +18.8 — 문헌이 예측하지 못하는 유일한 방향.**
   Gaze-Reg는 −7.4, 토큰 가지치기 계열도 전부 하락. 우리만 상승을 가진다.
   **이건 이상치가 아니라 논문의 중심에 놓아야 할 결과다.**
4. **픽셀 공간 foveation은 연산을 0% 아낀다.** VLA-Cache가 attention 마스킹에서
   같은 종류를 보였지만, 픽셀 예산과 토큰 예산의 분리를 **측정으로** 보인 것은
   우리가 처음으로 보인다(문헌 검색 범위 내).
5. **무엇이 먼저 죽는지는 무엇을 제거했는지에 달려 있다.** VLA-Pruner는
   "토큰을 semantic으로 자르면 action이 죽는다", 우리는 "용량을 자르면
   referential grounding이 죽고 motor는 산다". **반대 방향의 두 결과를 나란히
   놓는 것이 각각보다 강하다.**
6. **결정론적 폐루프에서의 paired 프로토콜.** PhAIL이 확률적 실로봇용 처방을
   냈고, 우리는 결정론 조건에서 더 강한 보장을 낸다.

### 5.3 논문 프레이밍 수정 제안

지금 우리 한 페이지는 *"부호가 뒤집힌다"*를 1번으로 놓고 있다. 문헌을 읽고 나면
**순서를 바꾸는 게 낫다:**

> **기존 연구는 각 개입을 하나의 벤치마크·하나의 백본에서 측정하고, 대개
> "성공률은 유지하면서 속도를 벌었다"로 보고한다. 우리는 같은 개입을 두
> 벤치마크 × 세 백본에서 에피소드 단위로 짝지어 재고, 그 결과 세 축 모두에서
> 효과의 부호가 측정 장소에 따라 뒤집힌다는 것을 보인다 — 그리고 시각 축에서는
> 아낀다고 알려진 연산이 실제로는 0이다.**

즉 **"기존 관행 → 우리 측정 → 뒤집힘"**의 순서다. 이러면 Related Work가
Introduction의 논거를 그대로 이어받는다.

---

## 6. 남은 확인 사항 (원문 접근이 되는 곳에서)

| 확인할 것 | 왜 중요한가 |
|---|---|
| **MoLe-VLA의 +9.7% vs +8%, 36.8% vs 5.6×** | 검색 스니펫끼리 수치가 다르다. 인용 전 확정 필수 |
| **Gaze-Reg의 78.5% / 85.9%의 정확한 조건** | foveation 강도(우리 keep 20%와 비교 가능한가), 백본, 시점 수 |
| **ShortGPT BI 정의의 정확한 수식과 calibration set** | 우리 "에피소드 첫 스텝 1회 보정"이 그들 절차와 얼마나 다른가 |
| **VLA-Cache가 FastV를 잰 하드웨어·구현** | "속도가 안 는다"가 구현 문제인지 원리 문제인지 |
| **PhAIL이 조사한 13편의 목록** | 우리 Related Work의 "N/검정 유무" 표를 그 목록으로 채울 수 있다 |
| **EfficientVLA(OpenReview SELYlDHZk2)** | 층 + 시각을 동시에 다루는 유일한 논문. 우리와 가장 가까운 설계 |

**접근 방법:** 이 세션은 egress 정책으로 논문 호스트가 전부 막혀 있다. 로컬
환경이나 다른 네트워크에서 PDF를 받아 `experiments/papers/`에 넣어주면 그때
직접 읽고 이 문서를 갱신하겠다.
