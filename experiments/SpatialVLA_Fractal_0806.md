# SpatialVLA / Fractal (Google Robot) — 2026-08-06

Bridge에서 나온 곡선(30.2 → 42.7 → 17.7, horizon 1·2·4)이 **벤치마크를 건너서도
성립하는가**를 보려고 시작했다. 로봇도, 액션 공간도, 정규화 통계도 Bridge와
다르므로 같은 정점이 나오면 "학습된 실행 길이" 가설이 크게 강해진다.

실행 환경: `/content/vsim/bin/python`,
`SpatialVLA/experiments/tome/tome_spatialvla_eval.py`, `spatialvla-4b-224-pt`,
`--policy-setup google_robot`, `--unnorm-key fractal20220817_data/0.1.0`,
EGL(Xvfb 불필요).

## 프로토콜

공개 표(SpatialVLA 논문 Table I)는 Google Robot을 **3열**로 보고한다 —
Pick Coke Can / Move Near / Open·Close Drawer. 우리는 앞의 두 열을 돌린다.
서랍은 레이트레이싱 셰이더에 113스텝이라 조건 하나에 3~5시간이고, 게다가
레퍼런스가 4 URDF × 9 스테이션 × 6 env = 216 에피소드로 훑는 것을 우리는 24개로
샘플링하게 되므로 공개 수치와 나란히 놓을 수도 없다. 범위에서 제외하고 그
이유를 논문에 적는다.

에피소드 수는 **각 태스크의 프로토콜 전체**다. 하네스가 직접 세고, caller는
숫자를 주지 않는다(`275c4bc`).

| task | 초기 상태 수 | 어디서 오는가 |
|---|---|---|
| coke can × 3 방향 | 25 | 물체 초기 xy 5×5 격자, `[-0.35,-0.12] × [-0.02,0.42]` |
| move_near | 60 | `episode_id` 0–59 = 물체 3종 세트 5 × (source,target) 6 × 배치 2 |

레퍼런스 스크립트(`DelinQu/SimplerEnv-OpenVLA`, SpatialVLA README가 지정한
평가 코드)와 대조한 설정값:

| | 레퍼런스 | 우리 |
|---|---|---|
| coke can env | `GraspSingleOpenedCokeCanInScene-v0` | 동일 |
| 방향 | `lr_switch` / `upright` / `laid_vertically` | 동일 |
| 물체 xy | `-0.35 -0.12 5` × `-0.02 0.42 5` | 동일 |
| move_near env | `MoveNearGoogleBakedTexInScene-v0` | **v0** (아래 참조) |
| move_near 범위 | `--obj-episode-range 0 60` | 동일 |
| robot init | coke 0.35/0.20, move_near 0.35/0.21 | `prepackaged_config`가 동일 값 |
| freq / steps | 3 / 513 / 80 | `prepackaged_config`가 동일 값 |

## baseline 결과

| task | 성공 | 공개 수치 |
|---|---|---|
| pick_horizontal_coke_can | 21/25 = 84.0% | |
| pick_vertical_coke_can | 22/25 = 88.0% | |
| pick_standing_coke_can | 21/25 = 84.0% | |
| **Pick Coke Can (3종 평균)** | **64/75 = 85.3%** | **81.0%** |
| **Move Near (v0)** | **50/60 = 83.3%** | **69.6%** |
| Move Near (v1, 참고) | 52/60 = 86.7% | — |
| **전체** | **114/135 = 84.4%** | |

ms/infer는 네 태스크 모두 892–908로 Bridge의 902와 같다. 같은 모델이 같은 일을
하고 있다.

### Pick Coke Can은 재현됐다 — 85.3 vs 81.0

4.3포인트 차이는 **URDF 표본 방식으로 설명된다**(아래). 로봇·컨트롤 모드·씬·
overlay·정규화 통계가 전부 맞았다는 뜻이고, Fractal 셋업이 정상이라는 확인은
이 열에서 받은 것으로 본다.

실패 4개가 어디서 났는지도 구조적이다. horizontal의 실패를 5×5 격자에 찍으면
(x가 음수일수록 로봇에서 멂):

```
        y=-0.02  +0.09  +0.20  +0.31  +0.42
x=-0.350   .       F      .      .      .
x=-0.293   F       .      .      .      F
x=-0.235   F       .      .      .      .
x=-0.178   .       .      .      .      .
x=-0.120   .       .      .      .      .
```

먼 쪽 세 줄에만 몰려 있고 가까운 두 줄은 10/10이다. 난수였다면 이렇게 몰리지
않는다 — 격자가 의도대로 걸렸다는 증거다.

### Move Near는 재현되지 않았다 — 83.3 vs 69.6

**14포인트 높다. 원인을 특정하지 못했다.** 시도한 것과 결과를 그대로 남긴다.

| 가설 | 검증 | 결과 |
|---|---|---|
| 60개 중 24개만 돌려서(정렬된 앞부분) 쉬운 세트만 봤다 | 60개 전부 실행 | 91.7 → 86.7, **5포인트만 설명** |
| 어려운 세트(콜라캔+레드불캔)에서 무너질 것 | 세트별 실패 집계 | 1/1/1/2/3으로 **고르게 분포**, 가설 기각 |
| 레퍼런스는 v0, 우리는 v1을 썼다 | v0로 60개 실행 | 86.7 → 83.3, **3.4포인트만 설명** |

남은 후보는 검증하지 않았다.

1. **URDF 순회 vs 추첨.** 레퍼런스는 4종(`None`,
   `recolor_tabletop_visual_matching_1/2`, `recolor_cabinet_visual_matching_1`)을
   바깥 루프로 명시 순회하므로 공개 수치는 **4 × 60 = 240 에피소드의 평균**이다.
   우리는 `prepackaged_config`가 episode RNG에서 뽑게 두는데, 시드 0–59로는
   10 / 22 / 12 / 16이 나온다(균등은 15씩). coke can의 +4.3도 같은 방식으로
   설명될 수 있다.
2. **정책 래퍼.** 우리 `SpatialVLAInference`는 `/content/SimplerEnv`에서 오고,
   공개 수치는 `DelinQu/SimplerEnv-OpenVLA`에서 나왔다. 두 레포의 래퍼가
   action ensemble 온도나 sticky gripper 설정에서 갈리면 그대로 점수 차가 된다.
   확인하려면 두 파일을 직접 대조해야 한다.

**이 문서의 move_near 숫자를 공개 69.6% 옆에 나란히 표에 올리면 안 된다.**
프로토콜이 다르다는 것을 알고 있고, 어디가 다른지는 부분적으로만 안다.

### 왜 그래도 진행하는가

우리 주장은 **같은 환경 안에서 baseline 대비 개입의 변화**이고, 짝지은 McNemar
검정은 절대값 오프셋에 영향받지 않는다. 세 조건이 같은 env·같은 시드·같은
에피소드 id를 재생하는 한, 14포인트의 공통 오프셋은 2×2 표의 어느 칸도 바꾸지
않는다.

공개 수치 대조의 목적은 "셋업이 망가지지 않았다"의 확인이고, 그 확인은
**Pick Coke Can에서 통과했다**. Move Near의 불일치는 논문에 각주로 남긴다.

이후 move_near는 **v0으로 통일한다** — 재현이 더 잘 돼서가 아니라(3.4포인트
차이는 결정적이지 않다) 레퍼런스 프로토콜이 v0이기 때문이다. v1 결과는
`results/spatialvla_fractal_0806/baseline_movenear_v1/`에 남겨둔다.

## 천장 효과 — 미리 알고 있어야 할 것

baseline이 84.4%다. Bridge는 30.2%였다.

| | Bridge | Fractal |
|---|---|---|
| baseline | 30.2% | 84.4% |
| 올라갈 여지 | +69.8 | **+15.6** |
| 내려갈 여지 | −30.2 | −84.4 |

**이득은 거의 못 재고 손해는 아주 잘 잰다.** Bridge에서 repeat 2가 +12.5를
냈지만, Fractal에서 같은 이득이 안 보여도 그것은 "가설이 틀렸다"가 아니라
"잴 수 없었다"다. 논문에서 이 구분을 하지 않으면 스스로 주장을 깎는다.

McNemar는 불일치쌍만 쓰므로 천장 근처에서도 손해 쪽 검정력은 유지된다. 실제로
Bridge의 repeat 4(−12.5)는 p=0.0501이었고 repeat 2→4(−25.0)는 p=3.9e-5였다.
**Fractal에서 기대할 수 있는 것은 "repeat 4에서 무너지는가"이지 "repeat 2에서
오르는가"가 아니다.**

## 다음

- [ ] action repeat 2 / 4를 같은 4 태스크(coke can 3종 + move_near_v0)로 실행.
      조건당 135 에피소드, ~65분.
- [ ] 짝지은 검정 후 Bridge 곡선과 대조.
- [ ] (선택) URDF를 추첨이 아니라 균등 순회하도록 바꿔 공개 수치와의 차이가
      줄어드는지 확인. 짝지은 비교에는 영향이 없으므로 우선순위는 낮다.
