# OpenVLA: Foveation × Retina × Chunk-exec 리포트

**날짜:** 2026-07-20
**세팅:** SimplerEnv WidowX-Bridge, N=24/task, `openvla/openvla-7b`,
`simple_eval.py`(`--model {openvla, openvla_foveated, openvla_foveated_blur,
openvla_retina, openvla_chunk}`)로 전부 같은 노트북/GPU 세션에서 측정.

---

## 1. baseline

| Task | Success (grasp) |
|---|---|
| Carrot | 16.7% (29.2) |
| Eggplant | 25.0% (54.2) |
| Spoon | 8.3% (12.5) |
| Stack | 12.5% (16.7) |
| **평균** | **15.6% (28.2)** |

## 2. Foveation: log-polar와 blur — 거의 동률

| Model | Carrot | Eggplant | Spoon | Stack | **평균** | Δ |
|---|---|---|---|---|---|---|
| **foveate (log-polar)** | 16.7 (50.0) | 33.3 (58.3) | 41.7 (70.8) | 45.8 (75.0) | **34.4%** | **+18.8pp** |
| **foveate (blur)** | 25.0 (41.7) | 62.5 (87.5) | 25.0 (45.8) | 20.8 (54.2) | **33.3%** | **+17.7pp** |

두 변형 모두 4개 태스크 전부 baseline 이상이며 평균도 거의 같다. ms/infer는
baseline·log-polar·blur 전부 ~511–522ms로 동일 — foveation은 latency
대책이 아니라는 사실이 네 번째 백본에서도 확인됐다.

**해석**: OpenVLA는 좌표 인코딩(SpatialVLA의 Ego3D), 압축 병목(RoboVLMs의
latent-query), 순환 상태(RoboVLMs의 LSTM) 중 아무것도 없다. 깨질 장치가
없으니 픽셀을 어떻게 왜곡하든(이동시키든 안 시키든) 순수하게 "주변부
노이즈 제거" 효과만 남고, 그래서 log-polar와 blur의 차이도 거의 없다 —
SpatialVLA(blur 압승)·UniVLA(log-polar 우세)와 대조되는 세 번째 패턴.

## 3. Retina — foveation + 시간적 캐싱 + 적응형 액션 재사용

`RetinotopicCachedOpenVLAInference`: log-polar foveation을 fovea/mid/outer
3개 링으로 나눠 화면 움직임이 임계값 아래면 캐시(이전 프레임) 재사용,
동시에 `_should_run_model`이 움직임·재사용횟수·액션크기 기준으로 모델
forward 자체를 건너뛰고 직전 액션을 재사용할지 결정하는 결합 기법.

| Model | Carrot | Eggplant | Spoon | Stack | **평균** | Δ success | Δ grasp |
|---|---|---|---|---|---|---|---|
| Retina | 4.2 (25.0) | 25.0 (45.8) | 16.7 (45.8) | 4.2 (20.8) | **12.5%** | **−3.1pp** | **+6.2pp** |

**예상 밖의 결과**: foveation 단독보다 훨씬 정교한 기법인데 baseline
보다도 나쁘다. 그런데 grasp rate는 baseline보다 오히려 오른다 — **잡는
것은 개선되는데 놓는(success) 것은 나빠지는, SpatialVLA의 기하 붕괴와
같은 시그니처**다.

### 가설과 진단 도구

`_should_run_model`은 `phase_info`(grasp 여부 포함)를 인자로 받지만
실제로는 한 번도 참조하지 않는다 — 순수하게 화면 프레임 차이만으로
재사용을 결정한다. 가설: 물체를 든 채 미세 조정하는 배치 단계는 그리퍼+
물체가 같이 움직여 프레임 간 변화가 작게 측정되고, 그래서 재사용
임계값을 자주 통과해 오래된 액션/캐시를 계속 쓰다가 정밀 배치에서
어긋난다.

이를 검증하기 위해 `openvla_inference.py`에 **grasp 전/후로 나눈
재사용률·캐시갱신률 진단 계측**을 추가했다(판단 로직 자체는 변경하지
않음, 순수 계측):
- `episode_stats()`에 `pregrasp_reuse_rate` / `postgrasp_reuse_rate`,
  `pregrasp_mean_refresh_ratio` / `postgrasp_mean_refresh_ratio` 추가
- 로컬 순수 로직 테스트(20 pre-grasp + 10 post-grasp 합성 시퀀스)로
  카운터 분리가 정확함을 확인
- **아직 실제 GPU에서 재실행하여 가설을 확인하지는 못했다** — 다음
  실험 후보.

## 4. chunk-exec: 적용 대상이 아님

OpenVLA는 forward당 7차원 액션 1개만 예측한다(`predict_action` 반환값
확인, 미래 다중 스텝 chunk head 없음). SpatialVLA/RoboVLMs/UniVLA의
chunk-exec은 "모델이 이미 계산해둔 미래 예측 중 일부를 재활용"하는
기법인데, OpenVLA에는 재활용할 미래 예측 자체가 없다. 동일 액션을
그대로 반복 실행(action-repeat)하는 구현(`ActionRepeatOpenVLAInference`)
을 만들어뒀지만, 이는 "이미 계산된 것을 쓰는" chunk-exec과 근본적으로
다른 메커니즘(맹목적 반복)이라 4-백본 비교표에서는 **"해당 없음"**으로
표기하고 실험에서 제외했다.

## 5. 결론

- foveation은 OpenVLA에서 변형과 무관하게 확실한 이득(+17~19pp) —
  좌표/압축/순환이 없는 백본의 "기준 사례".
- Retina는 예상과 달리 baseline보다 나쁘고, SpatialVLA와 유사한
  grasp-preserved/placement-fails 시그니처를 보인다 — 원인 진단
  계측을 추가했으나 실측 확인은 미완료.
- chunk-exec은 아키텍처 전제조건 미충족으로 "실패"가 아니라 "적용
  불가"로 기록한다.

## 재현

```bash
python simple_eval.py --model openvla --task widowx_carrot_on_plate --n-episodes 24
python simple_eval.py --model openvla_foveated --task widowx_carrot_on_plate --n-episodes 24 --foveated-keep-percent 20
python simple_eval.py --model openvla_foveated_blur --task widowx_carrot_on_plate --n-episodes 24 --foveated-keep-percent 20
python simple_eval.py --model openvla_retina --task widowx_carrot_on_plate --n-episodes 24
```
