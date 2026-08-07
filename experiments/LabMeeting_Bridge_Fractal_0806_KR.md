# 추론 시점 개입은 전이되지 않는다 — Bridge × Fractal, 2026-08-06

*(영문판: `LabMeeting_Bridge_Fractal_0806.md` — 두 문서는 같은 스크립트가 만든
같은 숫자를 쓴다.)*

**상태: 캠페인 진행 중.** `--` 표시는 아직 안 돌린 칸이다. 아래 모든 수치는
`python experiments/build_grid_report.py`가 `results/`의 에피소드별 기록에서
직접 생성한다. 옮겨적은 값은 하나도 없다. 조건이 끝날 때마다 다시 돌리면
표가 알아서 갱신된다.

---

## 무엇을 하려 했고, 무엇이 나왔나

값싼 추론 시점 개입 — 액션을 k 환경 스텝 유지, 관측을 foveate, 중복 디코더
레이어 우회 — 이 성공률을 안 깎으면서 wall-clock을 벌어준다는 걸 보이려 했다.
이건 method 주장이고, 성립하려면 그 효과가 **method의 속성**이어야 한다.

아니었다. 같은 코드를 같은 hook 지점에 넣고 네 백본 × 두 벤치마크로 돌리면
**효과의 부호가 바뀐다.** 크기가 아니라 부호다. 그래서 결론은 "우리 방법이
된다"도 아니고 "안 된다"도 아니다. **효율 개입을 백본 하나·벤치마크 하나에서
측정한 값은 다른 어디에서도 무엇이 일어날지 예측하지 못한다** — 이건 이 부류의
연구가 평가되는 방식에 대한 주장이고, 아래 표가 그 증거다.

---

## 그리드

성공률, 그리고 해당 열 자신의 baseline 대비 paired Δ.
`**` = p < 0.05, `***` = 이 캠페인의 Bonferroni 기준(α ≈ 0.003) 통과.
불일치 쌍만 쓰는 McNemar exact test이고, Δ는 **양쪽 실행에 모두 있는
에피소드만** 짝짓는다. 그래서 4개 중 3개만 끝난 조건은 baseline의 그 3개와
비교되지, 전체 프로토콜과 비교되지 않는다.

*기울임 +* `†` = **에피소드별 기록을 안 남긴** 이전 캠페인에서 측정된 칸.
unpaired이고, Δ가 이 표 첫 줄의 baseline이 아니라 **그 캠페인 자신의
baseline** 대비다. 그래서 이 열에서 다시 계산할 수 없고, 부호에 기대는
주장을 실어줄 수 없다.

| 조건 | OpenVLA<br>Bridge | OpenVLA<br>Fractal | SpatialVLA<br>Bridge | SpatialVLA<br>Fractal | UniVLA<br>Bridge |
|---|---|---|---|---|---|
| 원본 정책 | **15.6%** (n=96) | **38.5%** (n=135) | **30.2%** (n=96) | **84.4%** (n=135) | **78.1%** (n=96) |
| action repeat 2 | 7.3%  −8.3 | 29.3%  +9.3 | 42.7%  +12.5** | 84.4%  +0.0 | 7.3%  −70.8*** |
| action repeat 4 | 4.2%  −11.5*** | -- | 17.7%  −12.5 | 44.4%  −40.0*** | -- |
| foveation log-polar 20% | 34.4%  +18.8** | -- | *25.0%  −7.3†* | 85.2%  +0.7 | *86.5%  +8.3†* |
| foveation blur 20% | 33.3%  +17.7** | -- | *30.2%  −2.1†* | 86.7%  +1.3 | *76.0%  −2.1†* |
| depth prune 1 | 17.7%  +2.1 | -- | *22.9%  −9.4†* | -- | -- |
| depth prune 4 | 16.7%  +1.0 | -- | -- | -- | -- |

† legacy 칸과, 각각이 실제로 대비한 baseline:

| 백본 / 벤치마크 | 조건 | 성공률 | Δ | 자기 baseline | 출처 |
|---|---|---|---|---|---|
| SpatialVLA / Bridge | foveation log-polar | 25.0% | −7.3 | 32.3% | `SpatialVLA_Bridge_Grid.md` |
| SpatialVLA / Bridge | foveation blur | 30.2% | −2.1 | 32.3% | `SpatialVLA_Bridge_Grid.md` |
| SpatialVLA / Bridge | depth prune 1 (26개 중) | 22.9% | −9.4 | 32.3% | `SpatialVLA_Bridge_Grid.md` |
| UniVLA / Bridge | foveation log-polar | 86.5% | +8.3 | 78.1% | `ChunkExecFoveation_univla.md` |
| UniVLA / Bridge | foveation blur | 76.0% | −2.1 | 78.1% | `ChunkExecFoveation_univla.md` |

SpatialVLA/Bridge legacy의 baseline은 첫 줄의 30.2%가 아니라 **32.3%**다.
같은 정책, 같은 프로토콜, 다른 캠페인 — 같은 것을 두 번 잰 값이 2.1포인트
차이 난다. 그게 이 열에서 우리가 가진 노이즈 바닥에 가장 가까운 값이고,
**하필 그 칸들이 보고하는 blur 효과의 크기와 같다.**

OpenVLA/Bridge log-polar은 **paired다.** 에피소드 기록이
`RetinaBased/GoogleColab/results_reproduction_eager/`에 남아 있고, 그 캠페인의
baseline이 현재 baseline과 에피소드 단위로 완전히 동일하다(96/96). 스크립트가
빌려오기 전에 그 동일성을 검증한다.

---

## 이게 뒷받침하는 세 가지

### 1. 같은 개입이 백본에 따라 부호가 뒤집힌다

action repeat 2, 동일한 코드·동일한 hook 지점:

| 백본 | Δ | 불일치 | p |
|---|---|---|---|
| UniVLA | **−70.8** | 1 고쳐짐 / 69 깨짐 | < 1e-15 |
| OpenVLA | −8.3 | 3 / 11 | 0.057 |
| SpatialVLA | **+12.5** | 21 / 9 | 0.043 |

83포인트 스프레드에, 양 극단 둘 다 통계적으로 확립됐다. 더 나은 기본값을
찾으면 정리될 정도 차이가 아니다.

### 2. 정책을 그대로 두고 벤치마크만 바꿔도 부호가 뒤집힌다

| action repeat 2 | Bridge | Fractal |
|---|---|---|
| OpenVLA | −8.3 | **+9.3** (코크캔만, p = 0.19) |
| SpatialVLA | **+12.5** (p = 0.043) | **+0.0** (p = 1.00) |

같은 체크포인트, 같은 가중치, 같은 개입 — 벤치마크만 움직였다.
**어느 백본도 자기 Bridge 결과로 자기 Fractal 결과를 예측하지 못한다.**
OpenVLA는 부호가 뒤집히고, SpatialVLA는 이득이 사라진다.

*주의:* OpenVLA/Fractal repeat 2는 현재 코크캔 3종(135개 중 75개)만 반영돼
있다. `move_near_v0`이 아직 돌고 있고 숫자가 움직일 수 있다. **+9.3을 태스크
평균으로 인용하면 안 된다.**

### 3. 한 백본·한 벤치마크 안에서, 시간적 교란 ≫ 시각적 손실

SpatialVLA / Fractal, 같은 세션, 같은 프로토콜, 135개 전체:

| 개입 | 무엇을 제거하는가 | Δ | p |
|---|---|---|---|
| action repeat 4 | 4스텝 중 3스텝의 재계획 | **−40.0** | ≈ 0 |
| foveation log-polar 20% | 시각 샘플 밀도의 80% | +0.7 | 1.00 |
| foveation blur 20% | 시각 디테일의 80% (기하는 보존) | +1.3 ² | 1.00 |

**시각 정보를 5분의 4 지워도 보이지 않는다. 액션을 4스텝 유지하면 정책이
무너진다.** 여기서 유일하게 **벤치마크를 건너지 않는 주장**이다 — 정책 하나,
벤치마크 하나, 세션 하나 안에서 성립한다.

² blur는 135개 중 75개. `move_near_v0` 진행 중.

검출 한계: 불일치 15쌍이면 **±7포인트보다 큰 효과는 배제**된다. 여기서
"효과 없음"은 "그 크기의 효과는 없음"이지 0이 아니다.

---

## 가장 그럴듯한 설명, 그리고 그게 깨지는 지점

Bridge 결과 뒤에 자연스럽게 나온 메커니즘은 **학습된 실행 길이로부터의
거리**였다. OpenVLA(k=1)와 UniVLA(k=5)는 이미 학습된 길이에 배포돼 있어
움직이면 잃기만 한다. SpatialVLA는 유일하게 자기 action chunk **아래로**
배포된 백본이고, repeat 2에서 이득을 보는 것도 그것뿐이다. Bridge는 이 이야기로
정확히 설명된다.

**Fractal에서 깨진다.** 같은 체크포인트, 같은 chunk 크기인데 repeat 2가
+12.5 → +0.0이고 repeat 4는 −12.5 → −40.0이다. Bridge가 보여준 k=2의 정점이
아예 없다. 이득을 만드는 게 무엇이든 학습된 chunk 길이만은 아니다 — 그 값은
바뀌지 않았으니까.

**이건 숨기지 말고 발표해야 한다.** "가장 그럴듯한 메커니즘조차 벤치마크를
건너지 못한다"가 깔끔한 설명보다 논지의 더 강한 버전이다.

---

## 쫓아볼 만한 신호 하나

SpatialVLA / Fractal `pick_standing_coke_can`에서 **두 foveation 변종이
baseline 실패 4개를 정확히 똑같이 구제**했다 (에피소드 10, 17, 20, 23).
그러면서 깨뜨린 건 서로 다르다:

| | 실패 에피소드 |
|---|---|
| baseline | 10, 17, 20, 23 |
| log-polar | 22 |
| blur | 16 |

서로 무관한 두 화질 저하 — 하나는 픽셀을 이동시키고, 하나는 디테일만 지운다 —
가 같은 4개를 고치는 건 노이즈의 모양이 아니다. **그 에피소드들이 정보 부족이
아니라 정보 과잉으로 실패했을 가능성**을 시사한다. n=25에 4개 태스크 중
하나이므로 결과가 아니라 관찰이다. 다음 단계는 그 네 장면에 실제로 무엇이
있는지 들여다보는 것이다.

---

## 캠페인 현황

| | Bridge | Fractal |
|---|---|---|
| **OpenVLA** repeat {1,2,4} | 완료, paired | baseline + repeat 2 (4개 중 3개) |
| **SpatialVLA** repeat {1,2,4} | 완료, paired | 완료 |
| **OpenVLA** foveation | blur 완료, log-polar은 7월 캠페인 | 미시작 |
| **SpatialVLA** foveation | **unpaired legacy만 있음** | log-polar 완료, blur 4개 중 3개 |
| depth pruning | OpenVLA/Bridge만 | 미시작 |

### 부채 두 가지

**legacy 칸.** 기존 figure의 foveation/depth 네 칸(SpatialVLA −7.3 / −2.1,
UniVLA +8.3, RoboVLMs −19.8 / −16.7)은 에피소드별 기록을 안 남긴 캠페인에서
나왔다. n=96 unpaired이면 약 15포인트 이하는 해상도가 없다. **부호 반전 주장을
실어줄 수 없고**, 재측정하거나 빼야 한다.

**바닥과 천장.** SpatialVLA는 Fractal에서 84%, OpenVLA는 38%다. 바닥 근처의
+9와 천장 근처의 +9는 같은 성취가 아니다. 독자가 열 간 Δ를 무방비로 비교하지
않도록 **모든 Δ 옆에 baseline을 같이 적는다.**

### 다음 순서

1. OpenVLA repeat 2와 SpatialVLA blur의 `move_near_v0` — 4개 중 3개까지 간
   칸 둘을 마감한다.
2. **OpenVLA / Fractal action repeat 4** — 두 번째 백본에서 horizon 곡선
   (1, 2, 4)을 닫고, 주장 2가 딛고 선 2×2를 완성한다.
3. SpatialVLA / Bridge foveation을 에피소드 기록과 함께 재측정 — legacy 두 칸을
   paired test로 바꾼다. baseline이 이미 디스크에 있어 추가 비용이 없다.
4. baseline을 같은 세션에서 한 번 더 돌려 **에피소드별 노이즈 바닥**을 측정.
   지금 이 보고서의 모든 Δ는 알려지지 않은 바닥 위에서 읽히고 있다. log-polar은
   135개 중 15개를 뒤집으면서 순증이 0인데, 그 15개 중 몇 개가 개입 탓인지
   아직 말할 수 없다.

---

## 재현

```bash
python experiments/build_grid_report.py           # 위의 모든 표
python experiments/build_grid_report.py --json    # 기계 판독용

# 개별 비교 하나 — 2x2 일치표와 태스크별 상세까지
python adaptive_sparse_vla/paired_test.py \
  results/spatialvla_fractal_0806/baseline \
  results/spatialvla_fractal_0806/action_repeat4
```
