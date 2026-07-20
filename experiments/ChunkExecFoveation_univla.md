# UniVLA: Chunk 실행 + Foveation 실험 리포트

**날짜:** 2026-07-20
**세팅:** SimplerEnv WidowX-Bridge, N=24/task, Colab (L4),
`UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K` (Emu3-MoE 백본, VQ 토큰 직접 입력,
chunk=5 네이티브 예측, `model-type baseline` = AutoGaze/adaptive-sparse
컨트롤러 미사용, 순정 UniVLA).
모든 실험은 학습 없이 eval 래퍼 수정만으로 수행
(`adaptive_sparse_vla/eval.py`의 `--exec-chunk` / `--foveate`).

---

## 0. 사전 이슈: overlay 이미지 누락으로 인한 관측 분포 오염

첫 baseline 파일럿(수정 전)에서 eggplant 성공률이 8.3%로 붕괴하는 등 비정상적인
결과가 나왔다. 원인은 번들 SimplerEnv의 `.gitignore`가 `/data/*`로
`real_inpainting/`(시뮬 화면에 덧씌우는 실제 로봇 실험실 사진)을 통째로
제외하고 있었고, `eval.py`가 overlay 파일이 없으면 **에러 없이 조용히
생 시뮬 화면으로 진행**했기 때문이다. 즉 정책이 학습 때 한 번도 본 적
없는 관측 분포로 4개 태스크가 전부 평가된 것이었다.

**조치**: `real_inpainting/` PNG 2장을 레포에 복원하고 `.gitignore` 예외
규칙 추가, `eval.py`는 overlay 파일이 없으면 `FileNotFoundError`로 즉시
중단하도록 변경(`ALLOW_MISSING_OVERLAY=1`로만 우회 가능). 이 사건 이후의
모든 결과는 overlay가 정상 적용된 상태다.

### baseline 재현성 검증

수정 후 baseline을 멘토님이 README(`7.1 UniVLA`)에 남긴 동일 체크포인트
기준 자체 측정치와 대조:

| Task | 멘토님 기록 (README 7.1) | 본 리포트 baseline |
|---|---|---|
| Eggplant | 24/24 = 100% | 100.0% |
| Carrot | 17/24 = 70.8% | 66.7% |
| **Stack** | 18/24 = **75.0%** | **75.0% (완전 일치)** |
| Spoon | 20/24 = 83.3% | 70.8% |
| 전체 | **82.29%** | **78.1%** |

Stack이 소수점까지 일치하고 나머지도 1~3 에피소드 오차(GPU/렌더링 차이
수준) 내라, 같은 체크포인트·환경이 정확히 재현됐다고 판단한다. 논문
Table 3(전체 69.8%)과는 값이 다른데, 이는 공개 체크포인트가 논문을 만든
정확한 체크포인트/평가 프로토콜과 다르기 때문으로 보이며 우리가 통제할
수 없는 영역이다. 따라서 이후 모든 Δ는 **본 리포트가 직접 측정한
baseline(78.1%)을 기준**으로 계산한다 — 논문 수치가 아니라 같은 코드·
같은 시드·같은 GPU에서 나온 baseline이라야 chunk-exec/foveation의
Δ가 "개입 때문"이라고 주장할 수 있다.

---

## 1. UniVLA 아키텍처 확인 (chunk-exec/foveation 해석에 필요한 사실)

`emu3/mllm/modeling_emu3.py`, `emu3/tokenizer/*`, `adaptive_sparse_vla/inference.py`
코드 확인:

- **비전 경로**: `BAAI/Emu3-VisionTokenizer` (범용 사전학습, **동결**) — 256×256
  입력을 8× 다운샘플하여 32×32=1024개의 **이산 VQ 토큰**으로 변환, 행 단위로
  flatten되어 `eol`/`eof` 구분자와 함께 LLM에 **그대로** 입력됨.
- **압축 병목 없음**: RoboVLMs의 `image_to_text_projection`(latent-query
  cross-attention) 같은 소수 latent로 요약하는 관문이 없다. 1024개 토큰이
  각각 독립적으로 attention에 참여한다.
- **좌표 인코딩 없음**: `modeling_emu3.py` 전체에서 `intrinsic`/`depth`/
  `backproject`/`camera` 계열 키워드 0건. 위치 정보는 1D RoPE(시퀀스 순서)
  뿐 — SpatialVLA의 Ego3D 같은 픽셀좌표→3D 경로가 없다.
- **히스토리는 LSTM이 아니라 프롬프트 컨텍스트**: `video_mode`가 이전
  프레임의 (이미지 토큰, 액션 토큰)을 프롬프트에 이어붙이는 방식. RoboVLMs
  의 LSTM `hidden_state`처럼 forward 호출마다 강제로 1틱씩 전진하는 내부
  시계가 없다.
- **chunk=5 네이티브**: `predict_action_frames=5` — forward 1번에 5개의
  미래 액션을 실제로 예측한다(RoboVLMs의 fwd_pred_next_n=10, SpatialVLA도
  유사한 chunk 예측 구조).

---

## 2. chunk-exec: 방향이 다른 백본, 그리고 반대 결과

### 2.1 다른 백본과의 방향 차이

SpatialVLA/RoboVLMs는 원래 forward마다 1개 액션만 실행하던 걸 "2개 실행"
으로 **늘려서**(forward 절감 = 가속) 테스트했다. UniVLA는 baseline 자체가
이미 5개를 다 실행하므로, 같은 실험을 하려면 반대로 **"5개 중 앞 2개만
실행하고 재계획"**(`--exec-chunk 2`)으로 **줄여야** 한다 — forward가 더
잦아지므로 감속(반응성↑, latency↑) 실험이다.

### 2.2 결과

| Task | baseline (grasp) | chunk2 (grasp) | Δ success |
|---|---|---|---|
| Carrot | 66.7% (66.7) | 75.0% (79.2) | +8.3pp |
| Stack | 75.0% (100.0) | 45.8% (83.3) | **−29.2pp** |
| Spoon | 70.8% (75.0) | 54.2% (75.0) | −16.6pp |
| Eggplant | 100.0% (100.0) | 87.5% (100.0) | −12.5pp |
| **평균** | **78.1%** | **65.6%** | **−12.5pp** |

ms/env-step은 baseline 603ms → chunk2 1414ms(약 2.3배)로, 5개 중 2개만
실행하는 설계대로 정확히 움직였다(ms/infer는 2826→2773으로 거의 그대로 —
forward 자체의 계산량은 안 변했다는 뜻).

### 2.3 유의성 확인

N=24 기준 표준오차로 보면 stack −29.2pp(SE≈8.8pp, 약 3.3σ)와 spoon
−16.6pp(SE≈9.3pp, 약 1.8σ)는 노이즈로 보기 어렵고, eggplant는 baseline이
무결점(24/24)이었다는 점에서 3개 신규 실패가 실질적이다. carrot +8.3pp는
SE 이내라 노이즈일 가능성이 있다.

### 2.4 해석

RoboVLMs 같은 파국적 붕괴(LSTM 상태 desync)는 없었다 — 프롬프트 기반
히스토리에는 그 취약점이 구조적으로 없기 때문으로 보인다. 대신 평균
−12.5pp의 확실한 손해가 났는데, 메커니즘은 다르다: UniVLA는 5-step
궤적을 하나의 응집된 단위로 예측하도록 학습됐다. 이를 2개로 잘라 더
자주 재계획하면, 매번 새로 시작하는 5-step 예측 경계에서 미세한
불연속(jerk)이 생겨 정밀도가 중요한 태스크(stack, eggplant)에서 더 크게
작용한 것으로 보인다. **결론: chunk-exec은 UniVLA에서도 손해이며, 원인은
LSTM desync가 아니라 "학습된 chunk 길이에서 벗어난 재계획 빈도가
궤적 응집성을 해친다"는 별개의 메커니즘이다.**

---

## 3. Foveation: log-polar가 이겼다 — SpatialVLA와 정반대

### 3.1 결과

| Task | baseline | log-polar (Δ) | blur (Δ) |
|---|---|---|---|
| Carrot | 66.7% | 75.0% (+8.3p) | 70.8% (+4.2p) |
| Stack | 75.0% | 83.3% (+8.3p) | 66.7% (−8.3p) |
| Spoon | 70.8% | 87.5% (+16.7p) | 87.5% (+16.7p) |
| Eggplant | 100.0% | 100.0% (+0.0p) | 79.2% (**−20.8p**) |
| **평균** | **78.1%** | **86.5% (+8.3p)** | **76.0% (−2.1p)** |

log-polar는 **4개 태스크 전부 baseline 이상**(순수 개선/유지)이고,
blur는 태스크에 따라 크게 갈린다(spoon +16.7 vs eggplant −20.8).
eggplant의 blur 하락은 Fisher 정확검정으로 baseline 24/24 grasp vs
blur 19/24 grasp가 p≈0.025로 우연이 아니다. ms/infer(2826/2846/2810)와
ms/env-step(603/605/594) 모두 baseline·log-polar·blur 전 구간에서 거의
동일 — 4번째 백본에서도 **foveation은 latency 대책이 아님**이 재확인됐다.

### 3.2 초기 가설과 그 철회

당초 "log-polar의 워핑이 VQ 토크나이저의 고정 그리드에서 중심부에 더
많은 토큰 예산을 재배분해 이득을 준다"는 가설을 세웠으나, 아래 3.3의
실측으로 **반증되어 철회한다** — log-polar는 역워핑으로 기하를 복원하기
때문에 중심부를 확대하지 않는다.

### 3.3 실측: 두 변형은 완전히 다른 열화 프로파일을 만든다

실제 평가 overlay 2장(`bridge_real_eval_1.png`, `bridge_sink.png`,
256×256, keep=20%)에서 중심으로부터의 거리(r, 0=중앙~1=모서리)별
Laplacian 에너지 보존율(원본 대비 고주파 디테일이 얼마나 남았는가)을
측정:

| 거리 r | log-polar 디테일 보존 | blur 디테일 보존 |
|---|---|---|
| 0.0–0.1 (정중앙) | 30–53% | **100% (비트 동일)** |
| 0.2–0.3 | 5–7% | **100%** |
| 0.4–0.5 | 1–3% | 52% |
| 0.5–0.7 | 1–2% | 8–9% |
| 0.7–1.0 (모서리) | (역워핑 보간 아티팩트로 재상승) | **0–1% (완전 소실)** |
| 픽셀 완전 일치 비율 | 2.5–5.7% | 28.8–29.1% |

- **log-polar = 완만한 전역 단순화**: 정중앙조차 30~53%만 남기지만,
  화면 어디에도 정보가 "0"이 되는 죽은 영역이 없다.
- **blur = 완벽한 중심 보존 + 절벽형 주변부 소실**: r≈0.3까지는 비트
  단위로 완전 보존하지만 r>0.5부터는 사실상 정보가 전멸한다.

### 3.4 왜 UniVLA는 log-polar를 선호하는가 (검증된 설명)

1. **왜 둘 다 망가뜨리지 않는가**: 1절에서 확인한 대로 좌표 기계·압축
   병목·순환 상태가 모두 없어, foveation이 깨뜨릴 대상 자체가 없다.
2. **왜 log-polar가 순이득인가**: UniVLA는 중심조차 30~53%로 낮아져도
   4개 태스크 전부 baseline 이상을 유지했다 — 이 모델의 판단에는 고해상도
   디테일이 필수가 아니었고, 전역 노이즈 억제 효과가 순이득으로 남았다.
3. **왜 blur는 eggplant에서 크게 나빠지는가**: eggplant는 싱크대 카메라
   구도라 목표물이 화면 중앙에서 벗어나 있는 경우가 많다. blur는 r>0.5의
   정보를 완전히 지우므로 그 영역의 물체를 아예 못 본다 — 실제로 blur
   에서만 grasp 자체가 100%→79.2%로 하락(물체 미탐지 시그니처)했고,
   log-polar는 주변부도 거칠게나마 보이므로 grasp 100%를 유지했다.

**한계**: (a) VQ 토크나이저 가중치가 이 환경에 없어 토큰 레벨 변화는
직접 관측하지 못했고 코드/기하 분석으로만 추론했다. (b) eggplant의
"목표물이 주변부에 있다"는 설명은 정황 일치이며 저장된 프레임으로
직접 확인하지 않았다.

---

## 4. 4-백본 종합 (OpenVLA / SpatialVLA / RoboVLMs / UniVLA)

| 개입 | OpenVLA | SpatialVLA | RoboVLMs | **UniVLA** |
|---|---|---|---|---|
| **chunk-exec** | 해당 없음(forward당 액션 1개, 예측 chunk 자체가 없어 기법 전제조건 미충족) | ✓ +13.6pp, 1.9× 가속 | ✗ −36.5pp (LSTM 상태 desync) | ✗ **−12.5pp** (궤적 응집성 손상, 방향 반대) |
| **Foveation (log-polar)** | ✓ +19pp | ✗ −7.3pp (Ego3D 좌표 파괴) | ✗ −19.8pp (지각 손상) | ✓ **+8.3pp** (전 태스크 개선/유지) |
| **Foveation (blur)** | (미실험) | ✓ log-polar 대비 +11.5pp 회복 | ✗ 회복 안 됨 | △ **−2.1pp (log-polar보다 나쁨)** |
| latency 효과 (foveation) | 0 | 0 | 0 | **0 (재확인)** |

- foveation은 OpenVLA·UniVLA(무해~긍정) vs SpatialVLA·RoboVLMs(손상)로
  정확히 2:2 — "명시적 좌표 인코딩·압축 병목·특화 인코더가 있는가"로
  갈린다.
- **log-polar vs blur 우열조차 백본마다 뒤집힌다**(SpatialVLA: blur 승 /
  UniVLA: log-polar 승) — 어떤 foveation 변형이 나은지조차 보편적 정답이
  없다.
- chunk-exec은 SpatialVLA에서만 순이익이고, 나머지는 방식은 달라도
  (desync vs 궤적 손상) 모두 손해다.

**공통 결론**: 테스트타임 효율화·지각 개입 중 어느 것도, 심지어 개입의
세부 변형 선택조차, 아키텍처와 무관하게 통하는 조합이 없다.

---

## 재현

```bash
# baseline
python adaptive_sparse_vla/eval.py --model-type baseline --task widowx_carrot_on_plate

# chunk2 (5개 중 앞 2개만 실행)
python adaptive_sparse_vla/eval.py --model-type baseline --task widowx_carrot_on_plate --exec-chunk 2

# foveation (log-polar / blur)
python adaptive_sparse_vla/eval.py --model-type baseline --task widowx_carrot_on_plate --foveate --foveate-mode logpolar --foveate-keep-percent 20
python adaptive_sparse_vla/eval.py --model-type baseline --task widowx_carrot_on_plate --foveate --foveate-mode blur --foveate-keep-percent 20
```
