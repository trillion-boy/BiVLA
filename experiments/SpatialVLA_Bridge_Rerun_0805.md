# SpatialVLA / Bridge — 2026-08-05 재측정 (진행 중)

## 왜 다시 돌리는가

`SpatialVLA_Bridge_Grid.md`의 action repeat 2 결과(**+10.4**)가 우리 논문의
핵심 문장을 떠받치고 있는데, **그것만 짝지은 검정을 하지 않았다** (짝 안 지은
z=1.50, 유의하지 않음). 그런데 그 캠페인의 per-episode JSON이 남아 있지 않아
사후 검정이 불가능했다.

레포에 있던 SpatialVLA 결과(`results/vanilla_baselines/spatialvla_full_suite`,
19.8%)는 쓸 수 없다 — eggplant가 3/24로, overlay 이미지 누락 버그(`a25f3fd`
이전)의 지문이 그대로 남아 있다. 다른 세 태스크는 한두 에피소드 차이인데
eggplant만 3 vs 16으로 갈리는 것이 그 증상이다.

그래서 **baseline / action repeat 2 / action repeat 4 를 한 세션에서** 다시
측정한다. 두 조건만으로는 검정이 되고, 4를 더하면 가설까지 검증된다.

## 진행 상황

| 조건 | 상태 | 소요 |
|---|---|---|
| **baseline** | ✅ 완료, 커밋됨 | 92분 |
| action repeat 2 | ⬜ 대기 | ~50분 예상 |
| action repeat 4 | ⬜ 대기 | ~25분 예상 |

실행 환경: `/content/vsim/bin/python`,
`SpatialVLA/experiments/tome/tome_spatialvla_eval.py`,
`spatialvla-4b-224-pt`, `unnorm_key=bridge_orig/1.0.0`, EGL(Xvfb 불필요),
`SIMPLER_ENV_ROOT=/content/SimplerEnv`(온전한 설치본, overlay 정상).

## baseline 결과

4 tasks × N=24 = 96 에피소드. `results/spatialvla_bridge_0805/baseline/`.

| task | 성공 | grasp | ms/infer |
|---|---|---|---|
| eggplant | 14/24 = **58.3%** | 70.8% | 905 |
| carrot | 6/24 = **25.0%** | 45.8% | 906 |
| stack | 7/24 = **29.2%** | 58.3% | 899 |
| spoon | 2/24 = **8.3%** | 16.7% | 898 |
| **평균** | **29/96 = 30.2%** | **47.9%** | **902** |

### 옛 캠페인과의 대조 — 4개 중 3개가 정확히 일치

| task | 그리드 문서 | 08-05 재측정 |
|---|---|---|
| carrot | 25.0 | **25.0** ✅ |
| stack | 29.2 | **29.2** ✅ |
| spoon | 8.3 | **8.3** ✅ |
| eggplant | 66.7 | 58.3 (2 에피소드 차) |
| 평균 | 32.3% | 30.2% |

**그리드 문서의 baseline은 신뢰할 만한 값이었다.** 어긋난 eggplant는 24개 중
2개 차이이고, max_episode_steps가 120으로 다른 태스크의 2배라 경계선
에피소드가 가장 많은 태스크다. 같은 문서에 이미 기록된 SpatialVLA의 재현
흔들림(stack을 두 번 재서 33.3% / 29.2%)과 같은 크기다.

**다만 이후의 Δ는 모두 이 30.2% 기준으로 계산한다.** 짝지은 검정은
per-episode 대응이 필요하므로 옛 숫자와는 섞을 수 없다.

## 남은 두 조건에서 볼 것

핵심은 **eggplant가 어느 방향으로 움직이는가**다. 지금 58.3%.

| horizon | OpenVLA (확정) | SpatialVLA |
|---|---|---|
| 1 | 15.6% | **30.2%** |
| 2 | 7.3% (−8.3, p=0.057) | ? |
| 4 | 4.2% (**−11.5, p=0.0010**) | ? |

OpenVLA는 1→2→4에서 단조 감소했고 4에서 확정적으로 무너졌다(불일치쌍 11–0).
`UniVLA_Bridge_ActionRepeat.md`의 가설 — 손상은 절대 horizon이 아니라 **정책이
학습된 실행 길이로부터의 거리**를 따른다 — 은 SpatialVLA에 대해 정반대를
예측한다. 이 체크포인트의 processor config에 `action_chunk_size`가 실려 있고
학습 chunk가 ~4이므로, **repeat 4 근처가 정점**이어야 한다.

- **2와 4가 모두 baseline보다 높고 4가 정점** → 가설 지지. 같은 x축(모델
  호출당 env 스텝) 위에 OpenVLA의 하강 곡선과 SpatialVLA의 상승 곡선을 나란히
  놓을 수 있다.
- **2는 오르고 4는 떨어짐** → 정점이 2와 4 사이. 여전히 OpenVLA와 부호가
  반대이므로 핵심 주장은 성립한다.
- **둘 다 baseline 이하** → 옛 +10.4가 재현되지 않는다는 뜻이고, §5의 부호
  반전 주장을 철회해야 한다. 그 경우에도 결과 자체는 보고 가치가 있다.

## 이어서 돌리는 법

실행 셀을 그대로 재실행하면 `glob` 체크가 baseline 4개를 건너뛰고 repeat 2부터
시작한다. 단 `/content/results/spatialvla_0805/`가 살아 있어야 한다. 세션이
새로 뜬 경우 레포에서 복원할 수 있지만, **GPU가 바뀌었다면 baseline도 다시
돌려야 한다** — 짝지은 비교가 이 실험의 요점이기 때문이다.

```python
!nvidia-smi --query-gpu=name,uuid --format=csv,noheader
```

로 오늘 값과 대조할 것.
