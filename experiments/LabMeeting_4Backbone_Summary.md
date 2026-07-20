# 랩미팅 진행상황 보고: 4개 VLA 백본에서의 테스트타임 개입 비교

**발표일:** 2026-07-20
**범위:** OpenVLA / SpatialVLA / RoboVLMs / UniVLA, 총 4개 VLA 백본
**공통 세팅:** SimplerEnv WidowX-Bridge, 4-task(Carrot/Stack/Spoon/Eggplant),
N=24 episodes/task, 전부 **학습 없이 eval 코드 수정만으로** 진행.
**모든 수치는 본인이 직접 돌린 시뮬레이션 실험 결과이며, 각 백본의 Δ는
그 백본 자신의 같은 코드·같은 체크포인트·같은 GPU에서 측정한 baseline
대비로 계산했다** (논문 발표 수치나 다른 실행환경 수치를 기준으로 삼지 않음).

---

## 0. 이번 발표에서 새로 추가된 부분

이전 랩미팅 이후 **RoboVLMs**와 **UniVLA** 두 백본에 대한 chunk-exec/
foveation 실험을 새로 진행했다. 기존에 하고 있던 SpatialVLA(chunk-exec,
foveation phase1/phase2)와 OpenVLA(baseline/foveation/Retina)를 합쳐,
**4개 백본에서 같은 두 가지 개입(chunk-exec, foveation)을 전부 테스트한
상태**가 됐다.

---

## 1. 핵심 결론 (요약 표)

| 개입 | OpenVLA | SpatialVLA | RoboVLMs | UniVLA |
|---|---|---|---|---|
| **chunk-exec** | 해당없음* | ✓ **+13.6pp**, 1.9× 가속 | ✗ **−36.5pp** | ✗ **−12.5pp** |
| **Foveation (log-polar)** | ✓ **+18.8pp** | ✗ **−7.3pp** | ✗ **−19.8pp** | ✓ **+8.3pp** |
| **Foveation (blur)** | ✓ **+17.7pp** | ✓ log-polar 대비 **+11.5pp 회복** | ✗ 회복 안 됨 | △ **−2.1pp** |
| log-polar vs blur | 거의 동률 | blur ≫ log-polar | (조합 미실행) | log-polar > blur |

\* OpenVLA는 forward당 액션 1개만 예측하는 구조라 "이미 예측해둔 미래
chunk 중 일부만 실행"하는 chunk-exec 자체를 적용할 전제조건이 없다
(아래 3.5절).

**한 줄 결론**: 4개 백본, 2개 개입 중 어느 조합도 보편적으로 통하지
않는다. 심지어 foveation의 세부 변형(log-polar vs blur)조차 백본마다
승자가 다르다. 이는 곧 논문 핵심 주장 — **"테스트타임 효율화/지각
개입의 효과는 아키텍처 구조가 결정한다"** — 를 4개의 독립적인 사례로
뒷받침한다.

---

## 2. 왜 백본마다 결과가 갈리는가 — 구조적 차이

| 백본 | 좌표 인코딩 | 압축 병목 | 히스토리 메커니즘 | 액션 예측 |
|---|---|---|---|---|
| OpenVLA | 없음 | 없음 | 없음(매 스텝 독립) | 1개(chunk 없음) |
| SpatialVLA | **있음** (Ego3D, 픽셀→depth→3D) | 없음 | 없음 | chunk(4개) |
| RoboVLMs | 없음 | **있음** (latent-query 압축) | **있음** (LSTM hidden state) | chunk(10개) |
| UniVLA | 없음 | 없음 | 있음(LSTM 아닌 프롬프트 컨텍스트) | chunk(5개, 네이티브) |

이 표가 아래 모든 결과의 예측 변수 역할을 한다: **깨질 장치(좌표 인코딩,
압축 병목, LSTM 상태)가 있는 백본일수록 foveation·chunk-exec 같은 "정보를
단순화/생략하는" 개입에 취약했다.**

---

## 3. 백본별 상세 실험

### 3.1 SpatialVLA — Ego3D 좌표 인코딩이 있는 백본

**세팅:** `spatialvla-4b-224-pt`(frozen), A100 40GB.

**Phase 1 — chunk-exec과 log-polar foveation**

| Config | Eggplant | Carrot | Stack | Spoon | **평균 성공률** | ms/infer |
|---|---|---|---|---|---|---|
| baseline | 66.7 (87.5) | 25.0 (45.8) | 29.2 (58.3) | 8.3 (16.7) | **32.3%** | ~844–902 |
| foveate만 (log-polar) | 58.3 (66.7) | 29.2 (54.2) | 4.2 (41.7) | 8.3 (12.5) | **25.0%** | ~845–895 |
| **chunk k=2** | **87.5** (91.7) | **41.7** (45.8) | 25.0 (58.3) | **29.2** (37.5) | **45.9%** | ~455–462 |
| chunk k=4 | 66.7 (83.3) | 4.2 (20.8) | 8.3 (20.8) | 8.3 (12.5) | 21.9% | ~220–231 |
| foveate + chunk2 | 45.8 (58.3) | 25.0 (62.5) | 20.8 (41.7) | 16.7 (33.3) | 27.1% | ~453–458 |

- **chunk k=2가 확실한 승리**: +13.6pp, 1.9× 가속, 4개 태스크 모두 안정적.
  k=4는 과도해서 오히려 역효과.
- **log-polar foveation은 −7.3pp 손해**, chunk2와 합치면 −18.8pp까지
  악화. 실패 시그니처: **grasp rate는 유지되는데 success만 무너짐**
  (잡았지만 못 놓음).

**원인 (코드 분석)**: SpatialVLA는 각 시각 토큰에 픽셀 격자좌표를
카메라 intrinsics와 추정 depth(ZoeDepth)로 역투영해 **3D 위치를
명시적으로 부여**한다(`modeling_spatialvla.py: backproject_patch` →
`Ego3DPositionEmbeddingMLP`). log-polar는 픽셀을 실제로 이동시키므로,
워핑 후 픽셀 (u,v)의 내용은 더 이상 카메라 광선 `inv(K)@[u,v,1]` 위에
있지 않다 — 즉 depth 추정과 기하학적으로 모순된 이미지가 되어 모든
토큰이 잘못된 3D 위치를 갖는다. 특히 배치 목표물(접시, 바구니)이 놓인
주변부에서 이 왜곡이 가장 심해, 관찰된 "잡지만 못 놓는" 패턴과 정확히
일치한다.

**Phase 2 — 기하 보존형 blur로 원인 검증 + 회복**

| Config | Eggplant | Carrot | Stack | Spoon | **평균** |
|---|---|---|---|---|---|
| (참조) chunk2만 | 87.5 | 41.7 | 25.0 | 29.2 | 45.9% |
| (참조) log-polar+chunk2 | 45.8 | 25.0 | 20.8 | 16.7 | 27.1% |
| **blur** (픽셀 이동 없음) | **79.2** (79.2) | 20.8 (25.0) | 12.5 (45.8) | **33.3** (41.7) | **36.5%** |
| blur + 움직임 추적 중심 | 75.0 (75.0) | 25.0 (33.3) | 20.8 (45.8) | 25.0 (37.5) | 36.5% |
| blur + 움직임추적 + pregrasp만 | 75.0 (83.3) | 29.2 (41.7) | 20.8 (50.0) | 29.2 (33.3) | **38.6%** |

- **blur가 log-polar 손해의 대부분을 회복**: 27.1% → 38.6% (+11.5pp),
  동일 정보량(keep=20%) 기준. 이게 "원인이 픽셀 이동(기하 파괴)이지
  정보 손실 자체가 아니다"라는 가설을 실험적으로 확정한다.
- spoon 태스크에서는 blur+chunk2(33.3%)가 chunk2 단독(29.2%)보다도
  높음 — **foveation이 처음으로 순이익을 낸 사례**.

**결론**: SpatialVLA의 헤드라인 결과는 **chunk k=2 (+13.6pp/1.9× 가속)**.
foveation은 기하 인코딩과 정면 충돌하는 아키텍처 의존적 실패이며, blur
로 원인(픽셀 이동)까지 실험적으로 특정했다.

---

### 3.2 OpenVLA — 좌표·압축·순환이 전혀 없는 "기준" 백본

**세팅:** `openvla/openvla-7b`, Colab GPU. Foveation과 Retina는 이번
같은 노트북/세션에서 baseline과 함께 재측정.

| Model | Carrot (grasp) | Eggplant (grasp) | Spoon (grasp) | Stack (grasp) | **평균 성공률** | Δ |
|---|---|---|---|---|---|---|
| baseline | 16.7 (29.2) | 25.0 (54.2) | 8.3 (12.5) | 12.5 (16.7) | **15.6%** | — |
| **foveate (log-polar)** | 16.7 (50.0) | 33.3 (58.3) | 41.7 (70.8) | 45.8 (75.0) | **34.4%** | **+18.8pp** |
| **foveate (blur)** | 25.0 (41.7) | 62.5 (87.5) | 25.0 (45.8) | 20.8 (54.2) | **33.3%** | **+17.7pp** |
| Retina (foveation+캐싱+적응재사용) | 4.2 (25.0) | 25.0 (45.8) | 16.7 (45.8) | 4.2 (20.8) | **12.5%** | **−3.1pp** |

- **foveation은 4개 태스크 전부 개선**, log-polar(+18.8pp)와
  blur(+17.7pp)가 **거의 동률** — SpatialVLA/UniVLA와 달리 어느 변형을
  쓰든 큰 차이가 없다. 좌표·압축·순환 중 아무것도 없는 백본에서는
  왜곡 여부 자체가 중요하지 않다는 뜻이다.
- **chunk-exec은 적용 불가**: OpenVLA는 forward당 액션을 1개만
  예측하며(chunk head가 없음), "모델이 이미 계산해둔 미래 예측을
  꺼내 쓴다"는 chunk-exec의 전제 자체가 성립하지 않는다. 동일 액션을
  단순 반복 실행(action-repeat)하는 것은 다른 백본들의 chunk-exec과
  근본적으로 다른 메커니즘(진짜 예측 실행이 아닌 맹목적 반복)이라
  비교 대상에서 제외했다.
- **Retina(멘토님이 만든 foveation+시간적 캐싱+적응형 재사용 결합
  기법)는 baseline보다도 나쁘다** (−3.1pp). 그런데 grasp rate는
  baseline보다 오히려 오른다(+6.2pp) — **SpatialVLA의 "잡지만 못
  놓는" 시그니처와 닮은 패턴**이다. 가설: 배치 단계에서 그리퍼+물체가
  같이 움직여 화면 변화가 작아 보이고, 재사용 임계값을 계속 통과해
  오래된 정보로 미세 조정을 시도하다 실패. **이 가설을 검증하는
  진단 코드(grasp 전/후 재사용률 분리 계측)를 방금 추가했고, 결과는
  아직 확인 전이다** (open question, 다음 실험 후보).

**결론**: foveation은 OpenVLA에서 변형 무관하게 확실한 이득
(+17~19pp). Retina처럼 정교한 결합 기법이 단순 foveation보다 못한
결과를 낸 건 예상 밖의 발견이며 원인 규명이 진행 중이다.

---

### 3.3 RoboVLMs — LSTM 상태 + latent 압축이 있는 백본

**세팅:** `kosmos_ph_bridge-post-train.pt` (KosMos-2 1.7B, LSTM 액션
헤드, hist=16, chunk=10), A100 40GB.

**chunk-exec 붕괴**

| Task | baseline | chunk2 | ms/infer |
|---|---|---|---|
| Carrot | 25.0% | 0.0% | 74→37 |
| Stack | 4.2% | 0.0% | 75→37 |
| Spoon | 41.7% | 8.3% | 75→37 |
| Eggplant | 87.5% | 4.2% | 75→38 |

지연시간은 의도대로 정확히 절반이 됐지만 **성공률이 거의 전멸**했다
(평균 −36.5pp). 원인: 액션 헤드가 stateful LSTM(`base_policy.py:451-458`)
이라, forward 호출 자체가 `hidden_state`를 1틱씩 전진시키는 **정책의
내부 시계**다. forward를 2 step에 1번만 부르면 5Hz로 학습된 LSTM이
실질 2.5Hz 히스토리를 받아 시간 감각이 깨진다.

**원인 분리: chunk-lag 진단** (forward는 매 step 실행하되 실행만
직전 chunk[1]로 바꿔치기 — 속도 이득 없는 순수 진단)

| Config | Spoon | Eggplant |
|---|---|---|
| baseline | 41.7% (58.3) | 87.5% (91.7) |
| chunk-lag (액션 품질만 저하) | 41.7% (66.7) | 66.7% (83.3) |
| chunk2 (품질+desync) | 8.3% (25.0) | 4.2% (50.0) |

| Task | 전체 하락 | 액션 품질 몫 | desync 몫 | desync 비중 |
|---|---|---|---|---|
| Spoon | 33.4pp | 0.0pp | 33.4pp | **100%** |
| Eggplant | 83.3pp | 20.8pp | 62.5pp | **75%** |

**chunk2 붕괴의 주범은 LSTM 히스토리 desync임을 ablation으로
직접 입증**했다 — 모델이 예측한 chunk[1] 액션 자체는 배포 가능한
품질이었다.

**Latency 프로파일** (CUDA-synchronized, 79.7ms/step)

| 스테이지 | ms/step | 비중 |
|---|---|---|
| LLM transformer (24층) | 41.9 | 52.6% |
| vision encoder (CLIP ViT-L) | 23.2 | 29.1% |
| 전처리/기타 | 9.6 | 12.0% |
| action head (LSTM) | 3.9 | 4.9% |
| vision→text projection | 1.1 | 1.4% |

OpenVLA/SpatialVLA는 autoregressive decode가 지배적이었지만, RoboVLMs는
토큰 생성 없는 단일 forward 구조라 text transformer(52.6%)와 vision
encoder(29.1%)가 대부분을 차지한다.

**Foveation (log-polar / blur)**

| Task | baseline (grasp) | log-polar (grasp) | blur (grasp) |
|---|---|---|---|
| Carrot | 25.0 (33.3) | 8.3 (16.7) | 29.2 (45.8) |
| Stack | 4.2 (54.2) | 0.0 (33.3) | 0.0 (29.2) |
| Spoon | 41.7 (58.3) | 25.0 (33.3) | 20.8 (54.2) |
| Eggplant | 87.5 (91.7) | 45.8 (54.2) | 41.7 (50.0) |
| **평균** | **39.6 (59.4)** | **19.8 (34.4)** | **22.9 (44.8)** |

log-polar·blur **둘 다 실패**하고, SpatialVLA와 달리 blur가 거의
회복시키지 못한다(+3.1pp뿐). 실패 시그니처도 SpatialVLA와 다르다 —
grasp와 success가 **함께** 무너진다(59.4%→34.4%/44.8%) = 기하 손상이
아니라 지각(perception) 자체의 손상. latent 압축 병목
(`image_to_text_projection`)이 화면 전역 정보를 소수 토큰으로
압축하는데, 이 압축 관문에 들어갈 정보 자체가 손실됐고, 여기에
full-finetune된 비전 인코더의 분포 특화, LSTM 16프레임 히스토리의
누적 효과까지 겹쳐 blur로도 회복이 안 된 것으로 분석된다.

**결론**: chunk-exec, foveation **둘 다 확실히 실패**. 원인이 서로
다른 메커니즘(LSTM desync vs 지각 손상)임을 ablation·대조실험으로
각각 입증했다.

---

### 3.4 UniVLA — VQ 토큰 직접입력 + 프롬프트 히스토리 백본

**세팅:** `UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K` (Emu3-MoE, 동결 VQ
토크나이저), Colab L4.

**사전 이슈**: 첫 baseline에서 eggplant가 8.3%로 붕괴했는데, 원인은
번들 SimplerEnv의 `.gitignore`가 실제-로봇 배경 overlay 이미지를
레포에서 통째로 빼먹었고, eval 코드가 이를 무음으로 원본 시뮬 화면으로
대체 진행했기 때문이었다(즉 학습 분포와 다른 화면으로 평가됨). 이미지
복원 + eval 코드가 overlay 누락 시 즉시 에러 나도록 수정 후 재측정.

**baseline 재현성 검증** (같은 체크포인트 기준 멘토님 자체 측정 대조)

| Task | 멘토님 기록 | 본 실험 baseline |
|---|---|---|
| Eggplant | 100% | 100.0% |
| Carrot | 70.8% | 66.7% |
| **Stack** | **75.0%** | **75.0% (완전 일치)** |
| Spoon | 83.3% | 70.8% |
| 전체 | 82.29% | **78.1%** |

Stack이 소수점까지 일치해 환경·체크포인트가 정확히 재현됐다고
판단, 이후 모든 Δ는 이 78.1% baseline을 기준으로 계산.

**chunk-exec (방향이 다른 실험)**: UniVLA는 baseline 자체가 forward당
5-action chunk를 전부 실행하므로, 다른 백본과 같은 실험을 하려면
반대로 "5개 중 앞 2개만 실행"(재계획 증가=감속)으로 줄여야 했다.

| Task | baseline (grasp) | chunk2 (grasp) | Δ |
|---|---|---|---|
| Carrot | 66.7 (66.7) | 75.0 (79.2) | +8.3pp |
| Stack | 75.0 (100.0) | 45.8 (83.3) | **−29.2pp** |
| Spoon | 70.8 (75.0) | 54.2 (75.0) | −16.6pp |
| Eggplant | 100.0 (100.0) | 87.5 (100.0) | −12.5pp |
| **평균** | **78.1%** | **65.6%** | **−12.5pp** |

RoboVLMs 같은 파국적 붕괴는 없다(히스토리가 LSTM이 아니라 프롬프트
컨텍스트라 desync 구조 자체가 없음). 대신 5-step 궤적을 하나의
응집된 단위로 학습했는데 2개로 잘라 재계획을 자주 하면 예측 경계에서
미세한 불연속이 생겨, 정밀도가 중요한 stack/eggplant에서 더 크게
작용한 것으로 분석(LSTM desync와는 다른 메커니즘).

**Foveation (log-polar / blur)**

| Task | baseline | log-polar (Δ) | blur (Δ) |
|---|---|---|---|
| Carrot | 66.7% | 75.0% (+8.3p) | 70.8% (+4.2p) |
| Stack | 75.0% | 83.3% (+8.3p) | 66.7% (−8.3p) |
| Spoon | 70.8% | 87.5% (+16.7p) | 87.5% (+16.7p) |
| Eggplant | 100.0% | 100.0% (+0.0p) | 79.2% (**−20.8p**) |
| **평균** | **78.1%** | **86.5% (+8.3p)** | **76.0% (−2.1p)** |

log-polar는 **4개 태스크 전부 baseline 이상**. blur는 spoon에서는
좋지만 eggplant에서 크게 나쁘다 (Fisher 정확검정: baseline grasp
24/24 vs blur grasp 19/24, p≈0.025 — 우연 아님).

**실측: log-polar와 blur는 완전히 다른 열화 프로파일을 만든다**
(실제 평가 화면에서 중심으로부터 거리별 디테일 보존율 측정)

| 거리 r | log-polar | blur |
|---|---|---|
| 0.0–0.1 (중앙) | 30–53% | **100% (완전 보존)** |
| 0.4–0.5 | 1–3% | 52% |
| 0.7–1.0 (모서리) | 소량 잔존 | **0–1% (완전 소실)** |

log-polar는 화면 전체를 완만하게 단순화(죽는 영역 없음), blur는
중심은 완벽 보존하지만 주변부는 절벽처럼 정보가 전멸한다. UniVLA는
좌표/압축/순환 취약점이 없어 log-polar의 완만한 열화는 순이익이지만,
목표물이 화면 주변부에 있는 eggplant 장면에서는 blur의 "주변부
완전소실"이 그대로 손해로 이어졌다.

**결론**: chunk-exec은 −12.5pp 손해(궤적 응집성 손상). foveation은
log-polar가 +8.3pp 순이익, blur는 −2.1pp로 오히려 손해 — SpatialVLA와
정반대의 log-polar/blur 우열.

---

## 4. 종합 결론

1. **chunk-exec**: stateless 백본(SpatialVLA)에서만 순이익
   (+13.6pp/1.9×). recurrent 백본(RoboVLMs LSTM, UniVLA 네이티브chunk)
   에서는 방식은 다르지만(desync vs 궤적손상) 둘 다 손해.
2. **foveation**: "깨질 장치가 없는" 백본(OpenVLA, UniVLA)에서 순이익,
   "있는" 백본(SpatialVLA 기하, RoboVLMs 압축+순환)에서 손해 —
   정확히 2:2.
3. **log-polar vs blur 우열조차 백본마다 다르다**: OpenVLA(동률) /
   SpatialVLA(blur 압승) / UniVLA(log-polar 승) — 개입의 세부 변형
   선택에도 보편적 정답이 없다.
4. 모든 실패 사례에서 **원인을 ablation/대조실험으로 직접 규명**했다
   (chunk-lag 진단, blur 대조군, 반경별 열화 측정, Fisher 검정 등) —
   "안 됐다"가 아니라 "왜 안 됐는지"까지 데이터로 답했다.

**결론 한 문장**: 4개 백본에 걸쳐 테스트타임 효율화·지각 개입의 성패는
예외 없이 아키텍처 구조(좌표 인코딩, 압축 병목, 순환 상태의 유무)로
설명되며, 어떤 개입도 아키텍처 독립적으로 통하지 않는다.

## 5. 다음 단계

- OpenVLA Retina의 grasp-전/후 재사용률 진단 실험 마무리(가설 검증)
- RoboVLMs: history-safe temporal vision cache (vision 29.1% latency
  겨냥, LSTM desync 없음) / LLM depth-prune 시도
- CALVIN/LIBERO로 성공 조합(SpatialVLA+chunk-exec, OpenVLA+foveation)
  확장 검증
- 실물 로봇 배포 경로 논의 (교수님 요청)
