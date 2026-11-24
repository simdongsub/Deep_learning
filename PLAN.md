# 심층학습 기말 프로젝트 개발 계획

**기간** 2026년 11월 20일~30일, 총 11일 (한국 시간)  
**목표** CIFAR-10에서 동일한 ResNet-20 백본을 사용해 중간층 표현 결합의 효과를 검증하고, 11월 30일까지 재현 가능한 코드와 보고·발표 자료를 준비한다.

## 1. 과제 조건과 연구 설계

| 근거 | 반영 사항 |
|---|---|
| 교수님 안내서 1~4쪽 | 3~4인, 연구 질문 하나, 비교 변수 외 설정 고정, 동일 튜닝 예산, 핵심 3시드 평균 ± 표준편차, 요소별 Ablation, 오차율·파라미터 수·학습 시간, 표·그래프·해석, 4~6쪽 보고서·코드·발표. 영상처리 과목도 수강하면 같은 데이터셋과 같은 베이스라인으로 중복 제출하지 않는다. |
| 팀 제안서 1~6쪽 | CIFAR-10과 처음부터 학습한 ResNet-20, B0/B0+/F1~F4, 100/25/10%, 후보 LR 세 개, 3시드, FLOPs, 4종 Ablation. 연산 부족 시 25% 조건부터 줄인다. |
| 이번 구현 | 모델과 학습·재개·평가, 실험표 생성, 기본 분석, Colab 노트북을 제공한다. 팀은 본 실험, 그래프 해석, 논문 수치 비교, 보고서와 발표를 수행한다. Linear Probe·Welch 검정·혼동 행렬은 선택 후속 작업이다. |

Track A는 공식 안내상 사전학습 백본을 허용하며 팀은 전 모델을 처음부터 학습한다. 공식 학습률 통제를 우선해 **같은 데이터 비율의 여섯 모델에 공통 LR**을 적용한다. 비율별로 각 모델의 LR 후보를 모두 검증하고, 후보별 모델 6개의 최고 검증 정확도 평균이 가장 높은 LR을 고른다. 동률이면 작은 LR을 선택한다. 100%와 10%에서 LR이 달라지면 H3 해석에 이 차이를 명시한다. 팀은 첫 정식 실행 전 이 절과 예산, 동시 실행 슬롯 수를 PR에서 확정한다.

**연구 질문** 중간층과 마지막 층 Latent Vector를 결합하면 마지막 층만 쓰는 B0보다 오차율이 줄어드는가? 어떤 결합이 효과적이며 데이터 양에 따라 효과가 어떻게 달라지는가?

- H1. 계층적 표현 결합이 오차율을 낮춘다. 같은 비율에서 B0 대비 오차율 차이(%p)를 구하고 두 설정 중 큰 표본 SD를 판단 기준으로 사용한다. 개선폭이 이보다 작으면 과제 규칙에 따라 효과 없음으로 보고한다. 오차율이 커지면 악화 방향과 크기도 명시한다.
- H2. 학습형 F3/F4가 고정 F1/F2보다 유리하다. 같은 비율·시드의 F1~F4, B0+ 및 파라미터·FLOPs를 함께 비교한다. B0+는 가장 큰 F2에 맞춘 단일 통제군으로, 모든 결합과 각각 정밀하게 일치하는 설계는 아니다.
- H3. 저데이터에서 결합 이득이 크다. 같은 결합 모델에서 각 비율의 B0 대비 오차율 감소폭을 비교한다. 기울기 경로와 사영의 기여는 Ablation과 손실 곡선으로 검토한다. 25% 제외 시 중간 비율 결론을 보류한다.

### 공통 실험 설정

- split seed 2026으로 CIFAR-10 공식 학습 세트를 클래스당 학습 4,500장·검증 500장으로 분할한다. 학습은 100% 45,000장, 25% 11,250장, 10% 4,500장이고 부분집합은 서로 포함 관계다. 검증 5,000장과 공식 테스트 10,000장은 공통이다.
- 학습에만 RandomCrop(32, padding=4)과 RandomHorizontalFlip을 적용한다. 전 분할은 같은 채널 평균과 표준편차로 정규화한다. 반복 시드 0/1/2는 모델 초기화, 셔플 및 증강을 제어하고 분할에는 영향이 없다.
- 100 Epoch, 배치 128, SGD momentum 0.9 및 weight decay 0.0005, CosineAnnealingLR. 각 실행에서 최고 검증 정확도 체크포인트를 택하고 동률이면 처음 기록한 Epoch를 유지한다. 테스트는 LR·체크포인트·Ablation 대상을 고정한 뒤 최종 평가에 사용한다.
- 백본은 CIFAR용 20층, Stage별 16×32×32, 32×16×16, 64×8×8. GAP 후 S1/S2는 64차원 사영, S3는 64차원 그대로 사용한다. F2는 Concat 후 64차원 FC, F3는 Stage softmax 스칼라, F4는 입력별 채널 게이트를 Stage 축 softmax로 정규화한다. B0+의 64→120→64 잔차 FC는 F2 파라미터 수와 8개 차이다. 기본 설정의 파라미터 수는 B0 269,722, B0+ 285,266, F1 272,922, F2 285,274, F3 272,925, F4 275,050이다.
- B0 100% 3시드의 결과를 제안서의 He et al. 공개 오차율 약 8.75%와 비교한다. 목표는 ±1%p이며 벗어나면 에폭·분할·증강 등 프로토콜 차이를 기록한다.
- 정확도와 오류율은 %, 차이는 %p, SD는 표본 SD(분모 n−1), 학습 시간은 학습/검증 Epoch 구간의 누적 초를 보고한다. 별도로 Drive 저장·재시작까지 포함한 벽시계 시간도 실행표에 기록한다. FLOPs는 fvcore, 입력 1×3×32×32, FMA=1 기준과 미지원 연산을 함께 남긴다.

## 2. 개인별 PR 작업표

각 행의 담당자는 해당 소제목에 적힌 한 사람이다. 산출물에 `결과 경로`라고 적힌 항목은 Drive 또는 PR의 검증 증거로 남긴다. 기존 구현은 먼저 검증하고, 기준에 미달하는 부분을 해당 행의 파일에서 수정한다. 예상 시간은 작업 시간이며 학습 시간은 §6을 따른다.

### 동섭 · 환경, 학습 파이프라인, B0·B0+ 실행

| PR 단위 | 기한 | 생성·수정 파일 / 산출물 | 핵심 구현·실행 내용 | 선행 조건 | 검증과 완료 기준 | 예상 |
|---|---|---|---|---|---|---|
| DS-01 | 11/20(금) | `.gitignore`, `requirements.txt`, `README.md`, `data/.gitkeep`, `results/.gitkeep`, `checkpoints/.gitkeep` | Mac venv 설치 순서와 Colab CUDA 패키지 설치 순서 확인. main 보호 및 PR 검토 규칙 설정 | 기존 저장소와 origin 접근 | 새 체크아웃에서 import, `git check-ignore data/cifar-10-python.tar.gz checkpoints/x/last.pt .env` 확인 | 1시간 |
| DS-02 | 11/21(토) | `configs/default.yaml`, `dataset.py` | split seed와 실행 seed 분리, 클래스 균형 부분집합 및 train 전용 증강 확인 | DS-01 | `python dataset.py`, `python -m pytest -q -k split`. 학습 45,000/11,250/4,500, 검증 5,000, 공식 테스트 10,000 | 1.5시간 |
| DS-03 | 11/22(일) | `train.py`, `utils.py` | SGD/Cosine, Epoch별 train/val loss·accuracy, best 및 마지막 상태·RNG·시간 저장 확인. Epoch 중단 후 기존 ID로 이어서 실행 | DS-02, HH-01 | `python train.py --model B0 --ratio 10 --seed 0 --epochs 1 --run-id ds_smoke`. 별도 2 Epoch 실행에서 1 Epoch 저장 후 중단·`--resume`, CSV 중복 없음 | 2시간 |
| DS-04 | 11/23(월) | `colab.ipynb`, `README.md`, Drive smoke 로그 | GPU/Drive/고정 SHA 확인. 100%, 25%, 10%에서 각각 1 Epoch 실측하고 가용 GPU 시간과 저장 경로 전달 | DS-03 | CUDA 장치 출력, Drive의 `settings.json`·`history.csv`·`last.pt`, 새 런타임 재개 증거 | 1시간 + 실측 |
| DS-05 | 11/24~26 | Drive의 B0·B0+ 학습 결과 | 두 모델의 seed 0 LR 후보 3개를 비율별 실행. 공통 LR 확정 후 그 seed 0 재사용, seed 1/2 실행 | DS-04, HH-02, HS-02 | §6과 일치하는 실행 ID 및 설정. 전체 비율 기준 튜닝 18건, 핵심 신규 12건 | 실행 실측 |
| DS-06 | 11/27~28 | Drive의 `abl_s3_*`, `abl_s23_*`와 B0·B0+ 평가 | S3, S2+S3를 각 3시드로 학습. 자신의 핵심·Ablation 체크포인트를 평가해 `test.csv` 생성. B0 공개 오차율 8.75%와 비교 근거 전달 | DS-05, HS-03 | 변형 6행의 Stage 및 기준 ID 대조, B0 100%의 n=3 비교표 또는 누락 명시 | 실행 실측 |
| DS-07 | 11/30(월) | `README.md`, 재현 체크 기록 | 최종 SHA로 새 클론, 설치·Smoke·Drive 읽기 권한과 README 명령 검증 | DS-06, HS-06 초안 | 검증 SHA, 명령, 로그 링크를 제출 자료에 전달 | 1시간 |

### 환호 · ResNet-20·B0+·F1~F4 구현, F1·F2 실행

| PR 단위 | 기한 | 생성·수정 파일 / 산출물 | 핵심 구현·실행 내용 | 선행 조건 | 검증과 완료 기준 | 예상 |
|---|---|---|---|---|---|---|
| HH-01 | 11/21(토) | `models/__init__.py`, `models/resnet.py`, `model.py`, `tests/test_models.py` | CIFAR-10용 ResNet-20 Option A와 B0의 S3→GAP→10 logits 확인. `build_model`, `parameter_count` 사용 규격 전달 | 모델 입력 3×32×32 | `python -m pytest -q -k forward_backward`. Stage shape와 B0 269,722개 확인 | 2시간 |
| HH-02 | 11/22(일) | `model.py`, `models/fusion.py`, `tests/test_models.py` | B0+의 64→120→64 FC, F1 합, F2 concat→64 FC 확인. B0+/F2 8개 차이 검산 | HH-01 | `python -m pytest -q -k 'forward_backward or match'`. logits (2,10), 세 비교군 backward 통과 | 2시간 |
| HH-03 | 11/23(월) | `model.py`, `models/fusion.py`, `tests/test_models.py` | F3 Stage softmax와 F4 입력별 채널 게이트 검증. Stage 조합, pad, uniform, detach 검증 및 모델 코드 고정 | HH-02 | `python -m pytest -q`. F3 균등 합 1, F4 게이트 (N,S,64), detach 시 S3 경로 gradient 유지 | 2시간 |
| HH-04 | 11/24~26 | Drive의 F1·F2 학습 결과 | seed 0 LR 후보와 선택 LR의 seed 1/2 실행. 시드 0 선택 체크포인트 재사용 | HH-03, DS-04, HS-02 | 전체 비율 기준 튜닝 18건, 핵심 신규 12건의 이력·설정·체크포인트 대조 | 실행 실측 |
| HH-05 | 11/27~28 | Drive의 `abl_s13_*`, `abl_pad_*`와 F1·F2 평가 | S1+S3, 채널 패딩을 각 3시드로 실행하고 자신의 핵심·Ablation 평가 CSV 저장 | HH-04, HS-03 | 변형 6행에서 Stage 또는 사영만 변경됐는지 기준 ID별 확인 | 실행 실측 |
| HH-06 | 11/29(일) | 모델 구조 그림과 파라미터 표 | Stage별 채널·해상도, GAP/사영, F1~F4 결합 위치, B0+의 통제 대상을 그림·표로 전달 | HH-03, HH-05 | §1 기본 파라미터 수 및 코드와 같은 설명. W 모델과 각 Ablation 설정 표기 | 1시간 |

### 현성 · 실험표, 분석, 보고서·발표, F3·F4 실행

| PR 단위 | 기한 | 생성·수정 파일 / 산출물 | 핵심 구현·실행 내용 | 선행 조건 | 검증과 완료 기준 | 예상 |
|---|---|---|---|---|---|---|
| HS-01 | 11/20(금) | `PLAN.md`, 제안서 검토 기록 | H1~H3, 비교·반증 기준, 리소스 제한, 제출 항목 확인. 개인별 실행 ID와 담당자 확정 | 교수님 안내·팀 제안서 | 비교 변수, SD 기준, 선행 PR, 제출 담당 명시 | 1시간 |
| HS-02 | 11/23(월) | `experiments.py`, `evaluate.py`, `tune_plan.csv` | 계획 CSV의 소유자 검증, 후보 전체의 검증 로그로 공통 LR 선택, 최종 체크포인트 평가 형식 확인. 검증용 가상 로그는 임시 경로에서 사용 | DS-03. 실측은 DS-04, 모델 출력은 HH-03 의존 | 후보 54행과 동일 후보 수 확인, 기준 LR 선택과 seed 0 재사용 18행, 실제 Smoke에서 FLOPs·오차율 CSV 확인 | 2시간 |
| HS-03 | 11/26(목) | `ablation.py`, `selected_lr.json`, `core_plan.csv`, `ablation_plan.csv` | 24~26일 F3·F4 LR 후보와 seed 1/2 실행. 비율별 후보 로그 수집→공통 LR 확정→10% 검증 평균으로 W 선정→변형별 기준 ID와 소유자 배포 | HS-02, DS-04, HH-03. Ablation 기준 확정은 10% 핵심 로그 완료 후 | 전체 비율 기준 담당 튜닝 18건·핵심 신규 12건. Ablation 18행과 기준 ID 연결 확인 | 2시간 + 실행 |
| HS-04 | 11/28(토) | Drive의 `abl_uniform_*`, `abl_detach_*`와 F3·F4 평가 | 균등 α와 stop-gradient를 각 3시드로 학습. 자신의 핵심·Ablation 평가 CSV와 F4 게이트 저장 | HS-03 | 변형 6행의 비교 조건·F3 α·F4 게이트 기록 확인 | 실행 실측 |
| HS-05 | 11/28(토) | `analyze.py`, `results/analysis/` | 핵심 평균·표본 SD, 실험 시간·FLOPs, 손실 곡선, α·게이트, 시드별 Ablation 차이 생성. 누락 시 n과 실행 ID 확인 | HS-03. 최종 출력은 DS-06·HH-05·HS-04 결과 필요 | `python analyze.py results/core_plan.csv results/ablation_plan.csv`. 원본 CSV와 표·그래프 대조 | 2시간 |
| HS-06 | 11/29~30 | 보고서 4~6쪽, 발표 자료, 제출 확인표 | H1~H3 판정, B0와 공개 수치, 그림·표·Ablation·한계 작성. 최종 SHA와 결과 링크 확인 후 제출 준비 | HS-05, HH-06, DS-07 검증 증거는 30일 반영 | 모든 수치의 원본 경로 추적, 미완료 항목 표시, 발표 리허설 | 4시간 |

### AI에 PR 한 건을 맡길 때

아래 문장에서 PR ID 한 개만 지정한다. 기한, 파일, 전제, 검증 명령은 해당 행과 §6을 그대로 사용한다.

> PLAN.md의 **HH-03**을 수행해 주세요. 먼저 선행 PR의 코드와 테스트를 확인하고, 해당 행의 파일만 수정해 주세요. 기존 함수와 CLI 규격을 유지해 주세요. 명시된 검증 명령을 실행해 변경 파일, 실제 결과, 다음 PR에 전달할 경로를 알려 주세요. 실행에 필요한 로그·GPU·권한이 없으면 어떤 증거가 필요한지 보고하고 해당 작업을 미완료로 표시해 주세요.

GPU 실행 PR을 부탁할 때는 생성된 CSV에서 **실제 실행 ID 하나**, 고정 SHA, 결과·체크포인트 디렉터리를 같이 준다. 실행표의 체크포인트가 있으면 같은 설정으로 재개한다.

## 3. 날짜별 시작 순서

| 날짜 | 동섭 | 환호 | 현성 |
|---|---|---|---|
| 11/20 | DS-01 환경·Git | HH-01 백본 시작 | HS-01 가설·비교 설계 |
| 11/21 | DS-02 분할 검증 | HH-01 B0 규격 고정 | HS-02 실행표·평가 사전 검증 |
| 11/22 | DS-03 학습·재개 | HH-02 B0+·F1·F2 | HS-02 임시 로그로 LR 선택 검증 |
| 11/23 | DS-04 GPU 실측 | HH-03 F3·F4·Ablation 옵션 | HS-02 실행표 확정, 25% 유지 판단 |
| 11/24 | DS-05 100% B0·B0+ 탐색 | HH-04 100% F1·F2 탐색 | HS-03 100% F3·F4 탐색, 로그 수집 후 공통 LR 선택 |
| 11/25 | DS-05 100% 반복, 10% 후보 | HH-04 100% 반복, 10% 후보 | HS-03 10% 탐색 및 LR 확정, 여유 시 25% 큐 |
| 11/26 | DS-05 10% 반복, 유지 시 25% | HH-04 10% 반복, 유지 시 25% | HS-03 확정 비율 핵심 로그 마감, W·변형표 확정 |
| 11/27 | DS-06 S3·S2+S3 | HH-05 S1+S3·pad | HS-04 detach 우선, 비교 조건 고정 후 평가 |
| 11/28 | DS-06 평가 및 B0 비교 | HH-05 평가 자료 전달 | HS-04 균등 α 완료, HS-05 분석·그림 |
| 11/29 | DS-07 README 재현 점검 시작 | HH-06 모델 도식·파라미터 표 | HS-06 보고서·발표 초안 |
| 11/30 | DS-07 새 클론·Colab 검증 | HH-06 전달한 그림·수치 최종 점검 | HS-06 최종 자료·제출 확인 |

11/23에는 비율별 실측 Epoch 시간과 가용 슬롯을 비교한다. 초기 가정으로 핵심 고유 학습 90건은 30~45 GPU시간이다. GPU 두 개를 하루 6시간씩 사흘 쓸 경우 36시간이므로, 실측에 따라 23일 선행 실행 또는 §7의 비율 축소를 적용한다.

## 4. 교수님 공식 5단계 완료 체크리스트

| 확인 | 단계 | 담당 | 확인할 증거 |
|---|---|---|---|
| [ ] | 주제 제안 | 현성 | HS-01: 질문과 H1~H3, 데이터셋·트랙·반증 조건 |
| [ ] | 베이스라인 재현 | 동섭 | DS-03·04·06: B0 1 Epoch, Drive 저장·재개, 정식 B0와 공개 수치 비교 |
| [ ] | 중간 점검 · 모델 | 환호 | HH-03: 6개 모델 Forward/Backward와 파라미터, Ablation 옵션 |
| [ ] | 중간 점검 · 설계 | 현성 | HS-02: 고정 변인, 동일 튜닝, 실행표 |
| [ ] | 실험·분석 | 현성 | HS-05: 시드별 로그, 평균 ± SD, 학습 곡선, Ablation과 내부 가중치 |
| [ ] | 최종 발표 | 현성 | HS-06: 최종 표·그래프, 보고서, 발표 자료, DS-07의 재현 기록 |

## 5. GitHub 협업 절차

동섭은 GitHub 저장소 Settings의 브랜치 규칙에서 main 직접 푸시와 강제 푸시를 제한하고, PR과 리뷰 1명 이상을 요구한다. 테스트를 수동으로 수행해도 PR에 명령·결과를 남긴다. 논문 결과용 실행은 코드 변경이 합쳐진 SHA를 기준으로 한다. CSV 실행표의 단일 사본은 현성이 관리하고 담당자에게 행을 배정한다.

```bash
git status --short
git switch main
git pull --ff-only origin main
git switch -c feat/hwanho-hh-03
# 기능 단위로 수정 후 확인
git diff -- model.py models/fusion.py
python -m pytest -q
git add model.py models/fusion.py tests/test_models.py
git commit -m "Implement latent fusion classifiers"
git push -u origin feat/hwanho-hh-03
```

개인 브랜치 예시는 동섭 `feat/dongseop-ds-03`, 환호 `feat/hwanho-hh-03`, 현성 `feat/hyeonseong-hs-02`이다. 기존 브랜치는 새로 생성하는 대신 `git switch <브랜치>`로 이동한다. GitHub에서 base=main과 compare=개인 브랜치를 확인하고 PR에 변경 파일, 시험 방법, 로그 경로를 적는다. 리뷰어는 Files changed에서 파일별 diff를 보고 검증한다. 승인 후 머지하고 다음 작업 전 main을 다시 당긴다.

```bash
git fetch origin
git log --oneline HEAD..origin/main
git diff --name-status HEAD...origin/main
git diff origin/main...HEAD
git merge origin/main
# 충돌 시
git status --short
git diff --name-only --diff-filter=U
# 충돌 구간을 함께 검토하고 수정한 뒤 실제 수정한 파일을 add
git add model.py
python -m pytest -q
git commit -m "Merge main and resolve model interface conflict"
git push
```

작업 전 담당 파일을 합의하고 `model.py` 인터페이스 변경은 먼저 공유한다. 충돌에서 양쪽이 같은 파일의 API를 수정했다면 공동 검토 후 호출자와 테스트를 함께 고친다. 충돌 해결이 시작된 머지를 취소해야 하면 `git merge --abort`를 사용한다. 리뷰 중에는 큰 데이터, 토큰, 체크포인트가 diff에 없는지 확인한다.

## 6. 실험 실행과 GPU 예산

### 실행 ID 소유자

| 담당 | 핵심 비교와 LR 탐색 | Ablation |
|---|---|---|
| 동섭 | B0, B0+ | `s3`, `s23` |
| 환호 | F1, F2 | `s13`, `pad` |
| 현성 | F3, F4 | `uniform`, `detach` |

현성은 HS-02에서 핵심·튜닝 생성기의 `owner` 매핑을, HS-03에서 Ablation 생성기의 매핑을 위 표와 맞춘다. 생성기를 수정하기 전 실험표를 사용하면 이 표에 따라 소유자 열을 먼저 보정한다. 데이터 비율과 시드에 관계없이 같은 규칙을 적용하며, W가 어떤 모델이든 Ablation 소유자는 변형명으로 고정한다. GPU 슬롯에 문제가 생기면 실행 담당자가 동일 ID를 가능한 슬롯에서 이어서 처리한다.

### 실행 순서

1. 21일 smoke에서 CUDA 종류, 10%와 100% Epoch 시간, Drive 쓰기 시간, 런타임 한도를 기록한다. Sleep 또는 런타임 종료 중 재개가 필요하면 원래 목표 Epoch와 실행 ID를 계속 쓴다.
2. `python experiments.py tune --ratios 100 25 10`으로 후보 54행을 생성한다. 구간별로 진행할 때는 전체 후보표를 보존한다. 상태는 `planned → running → trained → evaluated`로 관리하고 중단은 `interrupted`, 범위 제외는 `dropped`로 표시한다. CSV의 `status`, `actual_seconds`는 담당자가 갱신한다. 생성 명령을 다시 실행하면 계획 CSV가 재작성되므로 상태표 사본을 먼저 저장한다.
3. 행마다 `python experiments.py run --output results/tune_plan.csv --run-id core_B0_r100_s0_lr0.05`처럼 실행한다. 중단 이후에는 같은 명령에 `--resume`을 붙인다. 실험표의 결과 폴더에서 `settings.json`, `history.csv`, 체크포인트 폴더에서 `last.pt`를 확인한다.
4. `python experiments.py select --ratios 100 25 10`으로 `selected_lr.json` 생성 후 `python experiments.py core --ratios 100 25 10`을 실행한다. 25% 제외 시 모든 관련 명령에서 `--ratios 100 10`을 일관되게 사용한다. 한 비율을 먼저 끝냈을 때 선행 실행표는 `--output`으로 별도 보존한다. `select`는 선택 파일을 덮어쓰므로 최종에는 유지할 비율 전체를 지정한다.
5. 핵심 CSV의 선택된 시드 0을 재사용하고 시드 1/2를 추가 학습한다. `python ablation.py --ratio 10`으로 기준 및 18행을 생성한다. 10%의 검증 평균 최고 F 모델을 기본으로 하되 균등 비교는 F3 대 F3-uniform이다. 다른 변형은 같은 기준 모델의 S1+S2+S3, 선형 사영, 기본 역전파 결과를 재사용한다.
6. 테스트 조건 확정 후 `python evaluate.py checkpoints/실행ID/last.pt --flops`. 분석은 `python analyze.py results/core_plan.csv results/ablation_plan.csv`. 미실행 Ablation 표가 없으면 핵심 표 하나만 넘긴다. Colab에서는 생성/평가 명령에 노트북의 Drive `--results-dir`, 생성/학습에 `--checkpoints-dir`도 맞춘다. 분석 출력은 `--output /content/drive/MyDrive/Deep_learning/results/analysis`를 지정한다.

| 구분 | 건수 | 초기 GPU 시간 (20~30분/회) |
|---|---:|---:|
| 제안서 기본 핵심 비교 | 54 = 6모델 × 3비율 × 3시드 | 18~27시간 |
| 제안서 Ablation | 18 = 6변형 × 3시드 | 6~9시간 |
| 튜닝 실행 전체 | 54 = 6모델 × 3비율 × 3 LR × 시드 0 | 18~27시간 |
| 튜닝과 핵심의 중복 | 18 = 선택된 시드 0 체크포인트 | 6~9시간 재사용 |
| 튜닝 순증분 | 36 | 12~18시간 |
| 전체 고유 학습 | **108**, 제안서 72회보다 36회 증가 | **36~54시간**, 제안서 24~36시간 대비 +12~18시간 |
| 25% 제외 시 | 핵심 36 + 튜닝 순증분 24 + Ablation 18 = **78** | **26~39시간** |

추정 시간은 계획값이다. 평가·저장·런타임 중단은 추가 시간이다. 각 행 실제 시간은 학습 종료 후 기록한다.

### 핵심 비교 실행표 · 54건

각 행의 ID에서 `LR비율`을 `selected_lr.json`의 숫자로 바꾼다. 예를 들어 100%의 공통 선택이 0.1이면 `core_B0_r100_s0_lr0.1`이다. 상대 저장 경로는 로컬 기준이며 Colab에서는 `results/`를 Drive의 results 경로로 치환한다. 각 실행은 별도 시드 단위로 기록한다.

| 실험 ID | 모델 | 데이터 비율 | Seed | LR | Epoch | 담당 | 상태 | 예상 시간 | 실제 시간 | 결과 저장 경로 |
|---|---|---:|---:|---|---:|---|---|---|---|---|
| `core_B0_r100_s0_lrLR100` | B0 | 100% | 0 | LR100 | 100 | 동섭 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_B0_r100_s0_lrLR100/` |
| `core_B0_r100_s1_lrLR100` | B0 | 100% | 1 | LR100 | 100 | 동섭 | 대기 | 20~30분 | — | `results/core_B0_r100_s1_lrLR100/` |
| `core_B0_r100_s2_lrLR100` | B0 | 100% | 2 | LR100 | 100 | 동섭 | 대기 | 20~30분 | — | `results/core_B0_r100_s2_lrLR100/` |
| `core_B0+_r100_s0_lrLR100` | B0+ | 100% | 0 | LR100 | 100 | 동섭 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_B0+_r100_s0_lrLR100/` |
| `core_B0+_r100_s1_lrLR100` | B0+ | 100% | 1 | LR100 | 100 | 동섭 | 대기 | 20~30분 | — | `results/core_B0+_r100_s1_lrLR100/` |
| `core_B0+_r100_s2_lrLR100` | B0+ | 100% | 2 | LR100 | 100 | 동섭 | 대기 | 20~30분 | — | `results/core_B0+_r100_s2_lrLR100/` |
| `core_F1_r100_s0_lrLR100` | F1 | 100% | 0 | LR100 | 100 | 환호 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F1_r100_s0_lrLR100/` |
| `core_F1_r100_s1_lrLR100` | F1 | 100% | 1 | LR100 | 100 | 환호 | 대기 | 20~30분 | — | `results/core_F1_r100_s1_lrLR100/` |
| `core_F1_r100_s2_lrLR100` | F1 | 100% | 2 | LR100 | 100 | 환호 | 대기 | 20~30분 | — | `results/core_F1_r100_s2_lrLR100/` |
| `core_F2_r100_s0_lrLR100` | F2 | 100% | 0 | LR100 | 100 | 환호 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F2_r100_s0_lrLR100/` |
| `core_F2_r100_s1_lrLR100` | F2 | 100% | 1 | LR100 | 100 | 환호 | 대기 | 20~30분 | — | `results/core_F2_r100_s1_lrLR100/` |
| `core_F2_r100_s2_lrLR100` | F2 | 100% | 2 | LR100 | 100 | 환호 | 대기 | 20~30분 | — | `results/core_F2_r100_s2_lrLR100/` |
| `core_F3_r100_s0_lrLR100` | F3 | 100% | 0 | LR100 | 100 | 현성 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F3_r100_s0_lrLR100/` |
| `core_F3_r100_s1_lrLR100` | F3 | 100% | 1 | LR100 | 100 | 현성 | 대기 | 20~30분 | — | `results/core_F3_r100_s1_lrLR100/` |
| `core_F3_r100_s2_lrLR100` | F3 | 100% | 2 | LR100 | 100 | 현성 | 대기 | 20~30분 | — | `results/core_F3_r100_s2_lrLR100/` |
| `core_F4_r100_s0_lrLR100` | F4 | 100% | 0 | LR100 | 100 | 현성 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F4_r100_s0_lrLR100/` |
| `core_F4_r100_s1_lrLR100` | F4 | 100% | 1 | LR100 | 100 | 현성 | 대기 | 20~30분 | — | `results/core_F4_r100_s1_lrLR100/` |
| `core_F4_r100_s2_lrLR100` | F4 | 100% | 2 | LR100 | 100 | 현성 | 대기 | 20~30분 | — | `results/core_F4_r100_s2_lrLR100/` |
| `core_B0_r25_s0_lrLR25` | B0 | 25% | 0 | LR25 | 100 | 동섭 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_B0_r25_s0_lrLR25/` |
| `core_B0_r25_s1_lrLR25` | B0 | 25% | 1 | LR25 | 100 | 동섭 | 대기 | 20~30분 | — | `results/core_B0_r25_s1_lrLR25/` |
| `core_B0_r25_s2_lrLR25` | B0 | 25% | 2 | LR25 | 100 | 동섭 | 대기 | 20~30분 | — | `results/core_B0_r25_s2_lrLR25/` |
| `core_B0+_r25_s0_lrLR25` | B0+ | 25% | 0 | LR25 | 100 | 동섭 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_B0+_r25_s0_lrLR25/` |
| `core_B0+_r25_s1_lrLR25` | B0+ | 25% | 1 | LR25 | 100 | 동섭 | 대기 | 20~30분 | — | `results/core_B0+_r25_s1_lrLR25/` |
| `core_B0+_r25_s2_lrLR25` | B0+ | 25% | 2 | LR25 | 100 | 동섭 | 대기 | 20~30분 | — | `results/core_B0+_r25_s2_lrLR25/` |
| `core_F1_r25_s0_lrLR25` | F1 | 25% | 0 | LR25 | 100 | 환호 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F1_r25_s0_lrLR25/` |
| `core_F1_r25_s1_lrLR25` | F1 | 25% | 1 | LR25 | 100 | 환호 | 대기 | 20~30분 | — | `results/core_F1_r25_s1_lrLR25/` |
| `core_F1_r25_s2_lrLR25` | F1 | 25% | 2 | LR25 | 100 | 환호 | 대기 | 20~30분 | — | `results/core_F1_r25_s2_lrLR25/` |
| `core_F2_r25_s0_lrLR25` | F2 | 25% | 0 | LR25 | 100 | 환호 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F2_r25_s0_lrLR25/` |
| `core_F2_r25_s1_lrLR25` | F2 | 25% | 1 | LR25 | 100 | 환호 | 대기 | 20~30분 | — | `results/core_F2_r25_s1_lrLR25/` |
| `core_F2_r25_s2_lrLR25` | F2 | 25% | 2 | LR25 | 100 | 환호 | 대기 | 20~30분 | — | `results/core_F2_r25_s2_lrLR25/` |
| `core_F3_r25_s0_lrLR25` | F3 | 25% | 0 | LR25 | 100 | 현성 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F3_r25_s0_lrLR25/` |
| `core_F3_r25_s1_lrLR25` | F3 | 25% | 1 | LR25 | 100 | 현성 | 대기 | 20~30분 | — | `results/core_F3_r25_s1_lrLR25/` |
| `core_F3_r25_s2_lrLR25` | F3 | 25% | 2 | LR25 | 100 | 현성 | 대기 | 20~30분 | — | `results/core_F3_r25_s2_lrLR25/` |
| `core_F4_r25_s0_lrLR25` | F4 | 25% | 0 | LR25 | 100 | 현성 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F4_r25_s0_lrLR25/` |
| `core_F4_r25_s1_lrLR25` | F4 | 25% | 1 | LR25 | 100 | 현성 | 대기 | 20~30분 | — | `results/core_F4_r25_s1_lrLR25/` |
| `core_F4_r25_s2_lrLR25` | F4 | 25% | 2 | LR25 | 100 | 현성 | 대기 | 20~30분 | — | `results/core_F4_r25_s2_lrLR25/` |
| `core_B0_r10_s0_lrLR10` | B0 | 10% | 0 | LR10 | 100 | 동섭 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_B0_r10_s0_lrLR10/` |
| `core_B0_r10_s1_lrLR10` | B0 | 10% | 1 | LR10 | 100 | 동섭 | 대기 | 20~30분 | — | `results/core_B0_r10_s1_lrLR10/` |
| `core_B0_r10_s2_lrLR10` | B0 | 10% | 2 | LR10 | 100 | 동섭 | 대기 | 20~30분 | — | `results/core_B0_r10_s2_lrLR10/` |
| `core_B0+_r10_s0_lrLR10` | B0+ | 10% | 0 | LR10 | 100 | 동섭 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_B0+_r10_s0_lrLR10/` |
| `core_B0+_r10_s1_lrLR10` | B0+ | 10% | 1 | LR10 | 100 | 동섭 | 대기 | 20~30분 | — | `results/core_B0+_r10_s1_lrLR10/` |
| `core_B0+_r10_s2_lrLR10` | B0+ | 10% | 2 | LR10 | 100 | 동섭 | 대기 | 20~30분 | — | `results/core_B0+_r10_s2_lrLR10/` |
| `core_F1_r10_s0_lrLR10` | F1 | 10% | 0 | LR10 | 100 | 환호 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F1_r10_s0_lrLR10/` |
| `core_F1_r10_s1_lrLR10` | F1 | 10% | 1 | LR10 | 100 | 환호 | 대기 | 20~30분 | — | `results/core_F1_r10_s1_lrLR10/` |
| `core_F1_r10_s2_lrLR10` | F1 | 10% | 2 | LR10 | 100 | 환호 | 대기 | 20~30분 | — | `results/core_F1_r10_s2_lrLR10/` |
| `core_F2_r10_s0_lrLR10` | F2 | 10% | 0 | LR10 | 100 | 환호 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F2_r10_s0_lrLR10/` |
| `core_F2_r10_s1_lrLR10` | F2 | 10% | 1 | LR10 | 100 | 환호 | 대기 | 20~30분 | — | `results/core_F2_r10_s1_lrLR10/` |
| `core_F2_r10_s2_lrLR10` | F2 | 10% | 2 | LR10 | 100 | 환호 | 대기 | 20~30분 | — | `results/core_F2_r10_s2_lrLR10/` |
| `core_F3_r10_s0_lrLR10` | F3 | 10% | 0 | LR10 | 100 | 현성 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F3_r10_s0_lrLR10/` |
| `core_F3_r10_s1_lrLR10` | F3 | 10% | 1 | LR10 | 100 | 현성 | 대기 | 20~30분 | — | `results/core_F3_r10_s1_lrLR10/` |
| `core_F3_r10_s2_lrLR10` | F3 | 10% | 2 | LR10 | 100 | 현성 | 대기 | 20~30분 | — | `results/core_F3_r10_s2_lrLR10/` |
| `core_F4_r10_s0_lrLR10` | F4 | 10% | 0 | LR10 | 100 | 현성 | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F4_r10_s0_lrLR10/` |
| `core_F4_r10_s1_lrLR10` | F4 | 10% | 1 | LR10 | 100 | 현성 | 대기 | 20~30분 | — | `results/core_F4_r10_s1_lrLR10/` |
| `core_F4_r10_s2_lrLR10` | F4 | 10% | 2 | LR10 | 100 | 현성 | 대기 | 20~30분 | — | `results/core_F4_r10_s2_lrLR10/` |

### Ablation 실행표 · 18건

`W`는 10%에서 시드 3개의 최고 검증 정확도 평균이 가장 높은 결합 모델이며 동률이면 F1→F2→F3→F4 순서를 쓴다. `LR10`은 10% 공통 학습률이다. 기준은 같은 시드의 원본 핵심 실행이다. S3 단독은 동일 결합 계열의 Stage 선택 통제를 위해 별도 실행하며, B0 수치는 참고로 병기한다.

| 실험 ID | 모델/변경 | 데이터 비율 | Seed | LR | Epoch | 담당 | 상태 | 예상 시간 | 실제 시간 | 결과 저장 경로 |
|---|---|---:|---:|---|---:|---|---|---|---|---|
| `abl_s3_W_r10_s0_lrLR10` | W · S3만 | 10% | 0 | LR10 | 100 | 동섭 | 대기 | 20~30분 | — | `results/abl_s3_W_r10_s0_lrLR10/` |
| `abl_s3_W_r10_s1_lrLR10` | W · S3만 | 10% | 1 | LR10 | 100 | 동섭 | 대기 | 20~30분 | — | `results/abl_s3_W_r10_s1_lrLR10/` |
| `abl_s3_W_r10_s2_lrLR10` | W · S3만 | 10% | 2 | LR10 | 100 | 동섭 | 대기 | 20~30분 | — | `results/abl_s3_W_r10_s2_lrLR10/` |
| `abl_s23_W_r10_s0_lrLR10` | W · S2+S3 | 10% | 0 | LR10 | 100 | 동섭 | 대기 | 20~30분 | — | `results/abl_s23_W_r10_s0_lrLR10/` |
| `abl_s23_W_r10_s1_lrLR10` | W · S2+S3 | 10% | 1 | LR10 | 100 | 동섭 | 대기 | 20~30분 | — | `results/abl_s23_W_r10_s1_lrLR10/` |
| `abl_s23_W_r10_s2_lrLR10` | W · S2+S3 | 10% | 2 | LR10 | 100 | 동섭 | 대기 | 20~30분 | — | `results/abl_s23_W_r10_s2_lrLR10/` |
| `abl_s13_W_r10_s0_lrLR10` | W · S1+S3 | 10% | 0 | LR10 | 100 | 환호 | 대기 | 20~30분 | — | `results/abl_s13_W_r10_s0_lrLR10/` |
| `abl_s13_W_r10_s1_lrLR10` | W · S1+S3 | 10% | 1 | LR10 | 100 | 환호 | 대기 | 20~30분 | — | `results/abl_s13_W_r10_s1_lrLR10/` |
| `abl_s13_W_r10_s2_lrLR10` | W · S1+S3 | 10% | 2 | LR10 | 100 | 환호 | 대기 | 20~30분 | — | `results/abl_s13_W_r10_s2_lrLR10/` |
| `abl_uniform_F3_r10_s0_lrLR10` | F3 · 균등 α | 10% | 0 | LR10 | 100 | 현성 | 대기 | 20~30분 | — | `results/abl_uniform_F3_r10_s0_lrLR10/` |
| `abl_uniform_F3_r10_s1_lrLR10` | F3 · 균등 α | 10% | 1 | LR10 | 100 | 현성 | 대기 | 20~30분 | — | `results/abl_uniform_F3_r10_s1_lrLR10/` |
| `abl_uniform_F3_r10_s2_lrLR10` | F3 · 균등 α | 10% | 2 | LR10 | 100 | 현성 | 대기 | 20~30분 | — | `results/abl_uniform_F3_r10_s2_lrLR10/` |
| `abl_pad_W_r10_s0_lrLR10` | W · 채널 패딩 | 10% | 0 | LR10 | 100 | 환호 | 대기 | 20~30분 | — | `results/abl_pad_W_r10_s0_lrLR10/` |
| `abl_pad_W_r10_s1_lrLR10` | W · 채널 패딩 | 10% | 1 | LR10 | 100 | 환호 | 대기 | 20~30분 | — | `results/abl_pad_W_r10_s1_lrLR10/` |
| `abl_pad_W_r10_s2_lrLR10` | W · 채널 패딩 | 10% | 2 | LR10 | 100 | 환호 | 대기 | 20~30분 | — | `results/abl_pad_W_r10_s2_lrLR10/` |
| `abl_detach_W_r10_s0_lrLR10` | W · 중간층 detach | 10% | 0 | LR10 | 100 | 현성 | 대기 | 20~30분 | — | `results/abl_detach_W_r10_s0_lrLR10/` |
| `abl_detach_W_r10_s1_lrLR10` | W · 중간층 detach | 10% | 1 | LR10 | 100 | 현성 | 대기 | 20~30분 | — | `results/abl_detach_W_r10_s1_lrLR10/` |
| `abl_detach_W_r10_s2_lrLR10` | W · 중간층 detach | 10% | 2 | LR10 | 100 | 현성 | 대기 | 20~30분 | — | `results/abl_detach_W_r10_s2_lrLR10/` |

## 7. 우선순위와 범위 조정

| 순위 | 지킬 작업 | 일정 판단 |
|---|---|---|
| P0 | CUDA 연결, CIFAR-10 분할, B0, 학습·평가, 시드·로그, Drive 재개 | 정식 GPU 큐 전에 모두 검증 |
| P1 | B0+, F1~F4, 세 시드·동일 튜닝·공통 LR, 결과표와 학습 곡선 | 방법론 30%, 해석 25%를 고려해 완전한 비교 묶음을 우선 확보 |
| P2 | 4종 Ablation, F3/F4 내부 가중치, Linear Probe, Welch 참고 검정, 혼동 행렬과 추가 그림 | 분석 배점에 직접 연결되는 detach·Stage, 균등·패딩 및 기본 α/게이트부터 수행. Probe·검정·추가 그림은 후순위 |

23일 저녁과 25일 종료 시 현성이 `남은 학습 수 × 비율별 실측`을 슬롯별로 계산한다. 모자라면 25% 전체를 먼저 제외하고 100%·10% 비교의 3시드와 공통 조건을 보존한다. 25% 일부를 이미 실행했다면 보조 로그에 남겨 최종 동일 조건 표에서 분리한다. 더 부족하면 100% 완전한 묶음 확보 및 10% 가능한 범위와 한계 보고를 우선하고 H3 판정을 보류한다. Ablation도 가능한 변형의 세 시드를 완결한다. 변경 날짜, 결정자, 이유, 예상/실측 시간, 영향 받은 실험 ID와 가설을 실행표 및 보고서에 적는다. 에폭·배치·시드·데이터 분할을 일부 모델만 바꾸거나 누락 값을 만들어 채우지 않는다.

## 8. 예상 문제와 해결 절차

| 문제 | 가능한 원인 | 확인 | 해결 절차 (담당) |
|---|---|---|---|
| Colab GPU 할당 실패 | GPU 할당량 소진 또는 CPU 런타임 | 런타임 유형, `nvidia-smi`, `torch.cuda.is_available()` | GPU 선택 후 재연결 → 담당자의 가용 슬롯으로 이동 → 동일 설정 재실행 (동섭) |
| 런타임 중단 | 유휴·시간 한도, 연결 해제 | Drive 마지막 `last.pt`, history Epoch | 새 런타임에서 동일 커밋/환경·Drive 연결 → 같은 ID `--resume` → 중복 Epoch 확인 (동섭) |
| GPU 메모리 부족 | 다른 GPU 프로세스, 메모리 단편화, 배치 용량 | `nvidia-smi`, CUDA OOM 위치 | 사용 안 하는 텐서/프로세스 정리 → 새 GPU로 재개 → 반복되면 전 비교군 공통 프로토콜 변경을 검토하고 기존 설정과 구분 (동섭) |
| CIFAR-10 다운로드 오류 | 서버·DNS·불완전 아카이브 | traceback, 다운로드 크기, torchvision 무결성 메시지 | 연결 재시도 → 정상 아카이브를 다른 팀 환경에서 확보 → 새 데이터 디렉터리에 복사 후 다운로드/무결성 재확인 (동섭) |
| PyTorch/CUDA 버전 충돌 | torchvision과 torch 불일치, CUDA 패키지 덮어쓰기 | 버전 출력, `nvidia-smi`, import traceback | 깨끗한 GPU 런타임에서 기본 CUDA 쌍 사용 → requirements만 설치 → Smoke (동섭) |
| 차원 오류 | Stage 순서, GAP 축, Concat 폭 | shape 로그, `pytest` | 표준 shape 16/32/64 확인 → 64차원 사영 후 결합 → logits (N,10) 및 backward 확인 (환호) |
| GitHub 머지 충돌 | 같은 줄 또는 공유 인터페이스 변경 | `git status`, `git diff --name-only --diff-filter=U` | 공유한 호출 규격과 양쪽 변경 확인 → 충돌 표지 제거·테스트 → 파일 add·commit·push (파일 담당) |
| 체크포인트 저장 실패 | Drive 미연결, 용량/권한 부족 | 쓰기 테스트, Drive 용량, `last.pt` 읽기 | 재마운트 → 경로/여유 확인 → 마지막 정상 파일에서 재개 → CSV Epoch 대조 (동섭) |
| 학습 과다 소요 | 100% 로더, 느린 GPU, Drive 쓰기 | 비율별 Epoch 중앙값, GPU 사용률, 저장 시간 | 병목 기록 → 이용 가능 슬롯 병렬 배치 → 25% 전체 축소 및 결정 기록 (동섭) |
| 재현 실패 | 커밋·버전·split seed·옵션 차이 | `settings.json`, SHA, 분할/시드, GPU 종류·로그 | 고정 커밋과 환경 재구성 → 같은 명령을 새 진단 ID로 실행 → 오차 원인과 동일 환경 재현 범위 기록 (동섭) |
| 성능 개선 미확인 | 시드 변동, Residual 정보 보존, 파라미터 증가 등 | 두 설정의 표본 SD, B0+, 손실 곡선, Ablation | 차이와 SD 및 방향을 수치로 보고 → α/게이트·detach 결과로 가능한 설명 검토 → 추정 설명을 한계로 표시 (현성) |

## 9. 분석 순서와 최종 제출 확인표

현성은 `summary.csv`와 `ablation_summary.csv`의 n을 확인하고 동일 비율·LR에서 시드별 오차율을 원본과 맞춘다. Ablation 차이는 변형 오차율 − 같은 시드 기준 오차율이며 양수는 변형에서 악화된 경우다. `*_loss.png`로 train/val 격차와 수렴을 비교한다. F3는 `*_alpha.png`와 검증 최고 시점의 α를 확인한다. F4는 `*_gates.png`와 `gates.csv`를 확인해 클래스별 채널 평균을 비교하고 시드 간 방향이 일치하는지 검토한다. 게이트 크기와 분류 개선의 인과 관계는 Ablation 근거와 함께 해석한다.

| 확인 | 제출 대상 | 담당 | 최종 확인 |
|---|---|---|---|
| [ ] | 제안서 | 현성 | Track A·질문·H1~H3·CIFAR-10/ResNet-20 일치 |
| [ ] | 정식 실험 계획/실행표 | 현성 | 54/18 및 튜닝 순증, 또는 축소 사유·누락 ID·학습 시간 표시 |
| [ ] | 보고서 4~6쪽 | 현성 | 연구 질문, 통제, 공개 B0 비교, 표·그래프·해석, 오차율·SD·효율, 한계 |
| [ ] | 발표 자료 | 현성 | 모델 도식, 핵심 그림, Ablation, 결론·한계, 리허설 |
| [ ] | GitHub 최종 코드 및 Colab | 동섭 | 고정 SHA, 새 클론, README 명령, CUDA Smoke와 Drive 재개 |
| [ ] | 재현 자료 | 동섭 | YAML, 시드·실행 ID·CSV·체크포인트 경로, Drive 읽기 권한, 결과 출처 |
| [ ] | 제출 준비 | 현성 | 공유 대상과 제출 위치 확인, 표/그래프/발표 수치가 같은 커밋과 결과를 가리킨다 |

### 계획과 저장소 대조

개인별 PR 표에는 기존 프로젝트 파일과 후속 산출물 경로를 적었다. `experiments.py`는 전체 학습을 자동 실행하지 않으며 각 ID는 §6의 소유자가 실행한다. 기본 분석은 최종 CSV·loss·α·F4 게이트 히트맵 및 Ablation 차이를 생성하고 F4 집계값은 `evaluate.py`가 기록한다. Linear Probe, Welch 검정, 혼동 행렬, 보고서와 발표 자료는 후속 일정에 배치했다.

### 스캐폴드 검증 기록

- Mac CPU에서 모델·Ablation·데이터 분할 자동 테스트 15개를 통과했다.
- 공식 CIFAR-10, 10%, 시드 0으로 B0 1 Epoch 학습 및 독립 테스트 평가를 완료했다. 학습·검증 구간은 약 20.8초였다. 이 실행은 동작 확인용이며 정식 비교표에 포함하지 않는다.
- 합성 입력으로 Epoch 중단 후 재개, 평가·FLOPs 저장, F3 α 및 F4 게이트 수집과 분석 출력을 확인했다. 검증 로그 픽스처로 공통 LR 선택·시드 0 재사용·실험표 건수를 확인했다.
- Colab 코드 셀의 Python 문법을 검증했다. 실제 Colab CUDA 및 Drive 연결·재개와 정식 학습은 팀 일정에서 확인한다.
