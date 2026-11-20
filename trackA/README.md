# Track A · 은닉층 Latent Vector 결합 설계

CIFAR-10 / ResNet-20 위에서 **결합 모듈만 바꿔** 오차율을 비교한다.
모든 설정은 같은 백본·같은 학습 조건을 쓰고, 바뀌는 것은 Head뿐이다.

## 구조

```
src/
  data.py              45,000/5,000/10,000 고정 분할, 100%/25%/10% 클래스 균형 부분집합
  train.py             학습·평가 엔트리포인트 (체크포인트 재개 포함)
  utils.py             시드, 파라미터 집계, config 병합
  models/
    resnet.py          ResNet-20 (Option A shortcut) · Plain-20, Stage 1/2/3 출력 반환
    fusion.py          B0 / B0+ / F1 더하기 / F2 Concat / F3 가중합 / F4 Gating
    net.py             백본 + Head 조립
configs/               설정 6종 + ablation/ 5종 + plain20
scripts/
  param_match.py       B0+ hidden 크기를 F2에 맞춰 계산
  make_jobs.py         튜닝 18 / 핵심 54 / Ablation 약 18회 명령 생성
  collect_results.py   summary.json 모아 평균 ± 표준편차 표 생성
```

## 사용

```bash
pip install -r requirements.txt

# 1) 베이스라인 재현 (목표: 8.75% ±1%p)
python -m src.train --config configs/b0.yaml --seed 0 --out runs

# 2) 파라미터 수 확인 + B0+ hidden 결정 → configs/b0plus.yaml 에 반영
#    (대상은 head 파라미터가 가장 많은 설정 — 현재 구현에서는 F4)
python -m scripts.param_match --target configs/f4.yaml

# 3) 학습률 탐색 {0.05, 0.1, 0.2} → 검증 세트로 선택
python -m scripts.make_jobs --stage tuning > jobs_tuning.sh && bash jobs_tuning.sh

# 4) 핵심 비교 54회 (선택된 lr)
python -m scripts.make_jobs --stage core --lr 0.1 > jobs_core.sh && bash jobs_core.sh

# 5) 결과 표
python -m scripts.collect_results --runs runs --out results
```

## 고정된 변인 (슬라이드 10)

| 항목 | 값 |
|---|---|
| 분할 | 45,000 / 5,000 / 10,000, SPLIT_SEED=12345로 1회 생성 |
| 증강 | RandomCrop(32, padding=4) + RandomHorizontalFlip |
| 최적화 | SGD, momentum 0.9, weight decay 5e-4 |
| 배치 · 에폭 | 128 · 100, Cosine 스케줄 |
| 튜닝 예산 | 모든 설정 동일하게 초기 lr {0.05, 0.1, 0.2}만 탐색 |
| 반복 | 시드 0 · 1 · 2, 평균 ± 표준편차 보고 |
| 해석 | 차이 < 표준편차 → "효과 없음" |

## Ablation 대응 (슬라이드 9)

| Ablation | 바꾸는 config 항목 |
|---|---|
| A1 결합하는 층 | `head.levels` — `[3]` / `[2,3]` / `[1,3]` / `[1,2,3]` |
| A2 가중치 학습 여부 | `head.learn_alpha: false` (F3) |
| A3 사영층 유무 | `head.proj: pad` (파라미터 없는 채널 패딩) |
| A4 기울기 전달 | `head.stop_grad: true` (중간층 경로만 detach) |

## 재현성

- 실행마다 `runs/<run>/` 에 `log.csv`(에폭별), `summary.json`, `config_resolved.yaml`,
  `ckpt_last.pt`, `ckpt_best.pt` 를 남긴다.
- 같은 `--out`·같은 설정으로 다시 실행하면 `ckpt_last.pt` 에서 이어서 학습한다
  (Colab 세션이 끊겨도 run을 버리지 않는다).
- `log.csv` 의 `alpha` 열에 F3의 α1~α3 추이가 기록된다 (슬라이드 13 분석용).
