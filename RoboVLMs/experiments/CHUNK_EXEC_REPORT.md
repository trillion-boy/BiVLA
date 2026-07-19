# RoboVLMs: Chunk 실행 + Foveation 실패 원인 진단 + Latency 프로파일 리포트

**날짜:** 2026-07-19
**세팅:** SimplerEnv WidowX-Bridge, N=24/task, A100 40GB,
`kosmos_ph_bridge-post-train.pt` (KosMos-2 1.7B, LSTM action head, hist=16, chunk=10).
모든 실험은 학습 없이 eval 래퍼 수정만으로 수행
(`eval/simpler/main_inference.py`의 `--exec-chunk` / `--chunk-lag-test` /
`--profile-latency` / `--foveate`).

---

## 1. 배경: chunk-exec k=2가 RoboVLMs에서 무너졌다

SpatialVLA에서 검증된 chunk 실행(모델이 이미 예측해 둔 미래 액션 chunk 중
k개를 연속 실행, forward는 k step에 1번)을 RoboVLMs에 이식한 결과:

| Task | baseline 성공률 | chunk2 성공률 | ms/infer |
|---|---|---|---|
| Carrot | 25.0% | 0.0% | 74→37 |
| Stack | 4.2% | 0.0% | 75→37 |
| Spoon | 41.7% | 8.3% | 75→37 |
| Eggplant | 87.5% | 4.2% | 75→38 |

지연시간은 의도대로 정확히 절반이 됐지만(2 step당 forward 1번) 성공률이
붕괴했다. SpatialVLA에서는 같은 개입이 +13.6pp/1.9배 가속이었다.

**구조적 원인 (코드 분석)**: RoboVLMs의 액션 헤드는 stateful LSTM이다
(`robovlms/model/policy_head/base_policy.py:451-458`). 매 forward 호출이
`hidden_state`를 정확히 1틱 전진시키고 `history_memory`에 프레임을 1개
축적한다. 즉 **forward 호출 자체가 정책의 내부 시계**다. forward를 2 step에
1번만 부르면 5Hz 연속 프레임으로 학습된 LSTM이 실질 2.5Hz 히스토리를 받아
시간 감각이 깨진다. SpatialVLA는 호출마다 독립(stateless)이라 이 문제가 없다.

## 2. 원인 분리 실험: chunk-lag 테스트

chunk2에는 잠재 원인이 두 개 섞여 있다:

- **(A) 액션 품질**: chunk의 t+1 액션(chunk[1]) 자체가 부정확할 가능성
- **(B) 히스토리 desync**: forward 건너뛰기로 인한 LSTM 상태 desync

`--chunk-lag-test`는 forward를 **매 step** 돌리고(B 제거) 실행만 직전
forward의 chunk[1]로 바꿔치기한다(A 유지). 속도 이득은 없고 순수 진단용.

### 결과 (성공률, 괄호는 grasp rate)

| Config | Spoon | Eggplant |
|---|---|---|
| baseline (A✗ B✗) | 41.7% (58.3) | 87.5% (91.7) |
| **chunk-lag (A만)** | **41.7%** (66.7) | **66.7%** (83.3) |
| chunk2 (A+B) | 8.3% (25.0) | 4.2% (50.0) |

### 하락폭 분해

| Task | 전체 하락 (baseline→chunk2) | A(액션 품질) 몫 | B(desync) 몫 | B 비중 |
|---|---|---|---|---|
| Spoon | 33.4pp | 0.0pp | 33.4pp | **100%** |
| Eggplant | 83.3pp | 20.8pp | 62.5pp | **75%** |

**결론: chunk2 붕괴의 주범은 LSTM 히스토리 desync다.**
모델이 예측하는 chunk[1] 액션 자체는 배포 가능한 품질이다 — spoon에서는
한 스텝 늦은 액션만 실행해도 baseline과 완전히 동일했다. eggplant의
A 몫 20.8pp는 에피소드가 120 step으로 길어 1-step 지연이 누적된 효과로
해석된다.

**논문 관점 시사점**: chunk-exec은 architecture-dependent하다.
stateless VLA(SpatialVLA)에서는 +13.6pp/1.9배의 공짜 점심이지만,
recurrent VLA(RoboVLMs)에서는 forward 호출을 줄이는 것이 곧 정책의
내부 상태 업데이트를 건너뛰는 것이 되어 실패하며, 그 원인이 액션 품질이
아니라 상태 desync임을 ablation으로 분리 입증했다. 이는 foveation이
SpatialVLA의 Ego3D 기하를 깨뜨렸던 발견과 대칭을 이루는, "테스트타임
효율화 기법은 백본 아키텍처와의 궁합이 결정한다"는 본 프로젝트 주장의
두 번째 근거다.

## 3. Latency 프로파일: 스텝당 79.7ms는 어디서 오나

`--profile-latency` (CUDA-synchronized, carrot 4 에피소드 = 240 step):

| 스테이지 | ms/step | 비중 |
|---|---|---|
| **LLM transformer (Kosmos2 text, 24층)** | **41.9** | **52.6%** |
| **vision encoder (CLIP ViT-L)** | **23.2** | **29.1%** |
| 전처리/토크나이즈/기타 (CPU) | 9.6 | 12.0% |
| action head (LSTM) | 3.9 | 4.9% |
| vision→text projection | 1.1 | 1.4% |
| **합계 (model.step)** | **79.7** | 100% |

(프로파일 중 ms/infer가 ~75→~80ms로 약간 커진 것은 CUDA sync 오버헤드,
비율에는 영향 없음.)

OpenVLA/SpatialVLA에서는 autoregressive decode가 지배적이었지만,
RoboVLMs는 토큰 생성이 없는 **단일 forward** 구조라 분포가 다르다:
text transformer가 절반, vision encoder가 약 30%를 차지한다.
LSTM 헤드는 4.9%로 거의 공짜 — 순환 구조의 계산 비용은 미미하지만,
그 순환성이 호출 횟수 절감(chunk-exec)을 막고 있다는 것이 핵심 아이러니다.

## 4. Foveation: log-polar와 blur 둘 다 실패했다

### 4.1 배경

Foveation(중심부 고해상도·주변부 저해상도로 압축)은 OpenVLA에서 +19pp,
SpatialVLA에서는 log-polar가 −7.3pp(Ego3D 위치 인코딩이 픽셀 좌표를 직접
쓰다 보니 왜곡에 기하가 깨져 grasp는 유지되고 배치만 실패)였고, 픽셀을
움직이지 않는 blur 변형이 +11.5pp를 회복시켰던 개입이다. RoboVLMs에서도
같은 4-task N=24 파일럿으로 두 변형 모두 재현했다
(`--foveate --foveate-mode {logpolar,blur} --foveate-keep-percent 20`,
`--foveate-center image`, `--foveate-phase always`). env는 원본 화면으로
스텝하고, 정책 입력만 foveate한다.

### 4.2 결과 (성공률, 괄호는 grasp rate)

| Task | baseline | log-polar | blur |
|---|---|---|---|
| Carrot | 25.0% (33.3) | 8.3% (16.7) | 29.2% (45.8) |
| Stack | 4.2% (54.2) | 0.0% (33.3) | 0.0% (29.2) |
| Spoon | 41.7% (58.3) | 25.0% (33.3) | 20.8% (54.2) |
| Eggplant | 87.5% (91.7) | 45.8% (54.2) | 41.7% (50.0) |
| **평균** | **39.6% (59.4)** | **19.8% (34.4)** | **22.9% (44.8)** |

ms/infer는 baseline(~75)과 log-polar/blur(~73–77) 모두 동일 — 예상대로
foveation은 latency 대책이 아니다(224×224 해상도가 그대로라 연산량 불변).

### 4.3 실패 시그니처가 SpatialVLA와 다르다

SpatialVLA는 grasp는 유지되고 success(배치)만 무너지는 기하 손상 특유의
패턴이었다. RoboVLMs는 **grasp와 success가 함께 무너진다**
(평균 grasp 59.4% → 34.4%(log-polar) / 44.8%(blur)) — 잡는 것부터
실패한다는 뜻으로, 기하가 아니라 지각(perception) 자체가 손상된 패턴이다.
blur가 log-polar 대비 회복시킨 폭도 평균 success +3.1pp뿐이라
(SpatialVLA에서의 +11.5pp 회복과 대조), 왜곡(log-polar) vs
비왜곡(blur)이라는 축이 RoboVLMs 실패의 주 원인이 아님을 보여준다.

### 4.4 원인: 왜곡 방식이 아니라 "정보 손실 그 자체"가 문제

log-polar와 blur는 픽셀 이동 여부만 다를 뿐 "중심 선명·주변 저하"라는
정보 손실 프로파일은 동일하다. 둘이 비슷하게 실패했다는 것은 원인이
SpatialVLA처럼 좌표계 파괴가 아니라 **주변부 정보 손실 자체**임을
시사한다. 유력한 메커니즘 3가지 (코드 근거):

1. `image_to_text_projection`의 latent-query cross-attention
   (`backbone.image_to_text_projection`, `robokosmos.py:40-44`)이 프레임
   전체를 소수의 latent 토큰으로 압축한다. 조작 태스크는 그리퍼·물체·목적지가
   화면 전역에 흩어져 있어(수건/바구니 등) 주변부 손실이 압축 단계의
   입력 정보 손실로 직결된다.
2. `train_setup.train_vision: True`로 CLIP ViT-L이 이 로봇 세팅의
   "깨끗한" 화면 분포(학습 시 `image_aug`는 shift만, 왜곡/블러 없음)에
   깊게 특화돼 있어, foveation은 학습 분포에 없던 강한 이탈이다.
3. LSTM 16프레임 히스토리(`window_size: 16`)가 매 스텝 동일하게 저하된
   신호를 상쇄 없이 누적한다 — chunk2를 죽인 것과 같은 순환 구조 취약점이
   여기서도 피해를 증폭시킨다.

### 4.5 결론

log-polar·blur 두 변형 모두 실패 — chunk-exec에 이은 세 번째
architecture-dependence 근거다. OpenVLA(+19pp, 깨질 좌표계·압축·순환
구조가 없음) / SpatialVLA(기하 붕괴, blur로 부분 회복) / RoboVLMs(지각
붕괴, blur로도 회복 안 됨)의 3-way 대비가 "테스트타임 효율화·지각 개입의
효과는 백본 아키텍처가 결정한다"는 프로젝트 주장을 완성한다.

## 5. 다음 단계 (Amdahl 상한 기준 우선순위)

1. **History-safe temporal vision cache** (vision 29.1% 겨냥, desync 없음):
   장면은 천천히 변하므로 vision encoder 출력을 2 step에 1번만 재계산하고
   **LLM+LSTM은 매 step 실행**. LSTM 상태가 매 step 전진하므로 chunk2를
   죽인 desync가 원천적으로 없다. SpatialVLA에서 stride-2 feature caching이
   성공률을 유지한 것과 같은 원리. 기대 절감 ~11ms/step (~14%).
2. **LLM depth-prune / layer-skip** (LLM 52.6% 겨냥):
   SpatialVLA에서 했던 redundant-layer 우회를 Kosmos2 text 24층에 적용.
   25% 우회 시 ~10ms/step (~13%) 절감 기대. 성공률 영향은 실측 필요.
3. **Foveation은 픽셀 왜곡이 아니라 토큰 축소로 재해석할 것**: 4절에서
   확인했듯 픽셀 레벨 foveation은 정보 손실 자체가 문제라 해상도를 유지한
   채로는 회복이 어렵다. vision 쪽 29.1% latency를 줄이려면 중심부 토큰은
   보존하고 주변부 vision token만 병합/제거하는 토큰 축소(ToMe/FastV류)가
   올바른 도구다.

## 재현

```bash
# chunk2 (4-task pilot)
python eval/simpler/main_inference.py ... --exec-chunk 2

# 원인 분리 진단 (forward 매 step + 직전 chunk[1] 실행)
python eval/simpler/main_inference.py ... --chunk-lag-test

# 스테이지별 latency 프로파일
python eval/simpler/main_inference.py ... --profile-latency

# foveation (log-polar / blur, 4-task pilot)
python eval/simpler/main_inference.py ... --foveate --foveate-mode logpolar --foveate-keep-percent 20
python eval/simpler/main_inference.py ... --foveate --foveate-mode blur --foveate-keep-percent 20
```
