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

## 2. 역할과 작업 인수인계

| 팀원 | 책임 파일과 업무 | 넘기는 자료 및 검토자 |
|---|---|---|
| A | `configs/default.yaml`, `dataset.py`, `train.py`, `utils.py`, `colab.ipynb`, 의존성·GitHub 환경, B0 및 Drive 재개 | B에게 입력·출력 규격, C에게 커밋·경로·B0 smoke 로그·실측 시간. B가 학습/분할 PR 검토 |
| B | `models/`, `model.py`, `tests/test_models.py`, B0+·F1~F4 및 기본 모델 테스트 | 모델명·파라미터 수·Ablation 옵션을 A/C에게 공유. A가 모델 PR 검토 |
| C | `experiments.py`, `ablation.py`, `evaluate.py`, `analyze.py`, 실행표·반복 실험·분석 | 검증 기반 LR, Ablation 기준 모델, 실행별 CSV, SD·α·게이트를 D에게 공유. A와 B가 설정 및 비교 검토 |
| D | `README.md`, `PLAN.md`, 표·학습 곡선 시각 검토, 해석·4~6쪽 보고서·발표 | 코드 SHA, 모든 표 수치의 원본 경로, 제출 확인표. C가 수치와 그림 검토 |

3인 팀은 C와 D를 통합한다. C가 실험·분석·보고 주담당을 맡고, A는 실행표 상태·README 재현 검증, B는 모델 도식과 Ablation 검토 및 발표 시연을 지원한다. GPU 슬롯이 여럿이면 CSV 담당자 기준으로 실행 ID를 서로 나눠 병행한다. 동일 ID는 담당 런타임 하나만 실행한다.

## 3. 날짜별 일정과 통과 조건

| 날짜 | 담당 | 순서와 작업 파일 | 당일 통과 기준 또는 다음 조치 |
|---|---|---|---|
| 11/20 금 | A/B/D | 기존 Git 루트 확인 → 팀 브랜치/PR 보호 원칙 → Mac 설치와 GPU 노트북 설정 → 연구 질문·가설 확정. A는 YAML/분할, B는 ResNet-20 틀 작성 | 팀별 브랜치와 리뷰 담당, 분할·모델 I/O 합의. GPU 미할당 시 21일 다른 팀원 슬롯 지정 |
| 11/21 토 | A/B | `dataset.py` 분할 테스트 → B0 Forward → Colab에서 10% 1 Epoch → Drive 파일 확인 → 동일 명령 재개 → 100%에서도 Epoch 시간 측정 | CUDA·손실·정확도·CSV·last.pt 확인. 논문 기준을 D에게 전달. GPU 시간으로 25% 유지 결정 초안 |
| 11/22 일 | A/B/C | A는 정식 B0·재개, B는 B0+·F1·F2, C는 `experiments.py tune` CSV와 시드 체크 | Forward/Backward와 파라미터 차이 8개 검산. B0 시드 0 탐색부터 시작 가능 |
| 11/23 월 | A/B/C | F3/F4·Ablation 옵션 → 전체 모델 테스트 → FLOPs 및 로그 형식 검증 → 커밋 고정 → 시드 0 탐색 선행 실행 | 실행별 담당자, GPU 슬롯, 공통 조건, CSV 경로를 미리 확정. 이 날 팀 GPU 용량으로 25% 포함 여부 고정 |
| 11/24 화 | A/B/C | 100% LR 18건 병행 → 완료 시 `select --ratios 100`으로 공통 LR 확정 → 별도 ID로 시드 1/2 큐 준비 | 후보 누락 없이 검증 로그로만 선택. 남은 GPU 시간과 10% 진행 배정 확인 |
| 11/25 수 | A/B/C | 100% 시드 1/2 계속 → 10% 탐색 18건 및 시드 1/2, 여유 시 예약한 25% 탐색 병행 | 우선 100%와 10%의 모델별 3시드 확보. 재사용 Seed 0 중복 실행 여부 점검 |
| 11/26 목 | A/B/C/D | 선택 비율 탐색 마감 → 선택 비율 전체로 `select`와 `core` 재생성 → 25% 시드 1/2 완료 목표 → C가 10% Ablation 기준 선정 | 최종 실험표, 고정 LR, 설정·커밋·실행 상태 확인. 미완료 건과 이유 표시 |
| 11/27 금 | C/B/D | `ablation.py`로 변형 6종 큐 생성 → 우선 detach 및 Stage 선택 → 핵심 비교 테스트 CSV 수집 | 변형별 원본 ID 연결 확인. 모든 조건 동결 후 테스트 실행 |
| 11/28 토 | C/B/D | 균등·패딩 Ablation 및 평가 → `analyze.py` → SD·곡선·F3 α·F4 클래스 게이트 검토 | n과 누락 ID 확인, 비교 불가 셀 표시. 여유가 있으면 Probe·Welch·혼동 행렬 |
| 11/29 일 | D/C/A | 결과표와 가설 판정 → 4~6쪽 보고서 작성 → 발표 슬라이드 및 README 실행 검토 | 표의 수치·단위·시드 수 추적, 부정적 결과와 한계 반영 |
| 11/30 월 | 전원 | 새 클론에서 설치·테스트·Colab Smoke → 제출 커밋·Drive 접근 검증 → 보고서·발표 리허설 및 제출 준비 | 마감 확인표와 실험 상태를 최종 커밋/공유본과 대조 |

GPU 시작·중단을 감안해 23일에도 튜닝을 선행한다. 초기 회당 20~30분 기준으로 24~26일 핵심 90개 고유 학습은 30~45 GPU시간이다. 하루 6시간씩 확보하는 GPU 두 개는 3일에 36시간을 제공하므로, 100% 실측이 높으면 선행 또는 비율 축소가 필요하다. 실제 배정은 각 비율의 중앙 Epoch 시간 × 남은 Epoch로 다시 계산한다.

### 교수님 공식 5단계 완료 체크리스트

체크는 증거 파일을 링크할 수 있을 때 팀원이 직접 갱신한다.

1. **주제 제안** (20일, D)  
   - [ ] Track A, 데이터셋, 질문 하나, H1~H3와 반증 조건을 팀 PR로 확정했다.
2. **베이스라인 재현** (21~24일, A/B)  
   - [ ] ResNet-20 Forward, 출력 (2,10), B0 실제 데이터 1 Epoch 검증을 확인했다.
   - [ ] Colab CUDA, Drive 로그·체크포인트 및 런타임 재연결 후 재개를 확인했다.
   - [ ] B0 정식 평균과 공개 수치 차이를 기록했다.
3. **중간 점검** (22~23일, A/B/C)  
   - [ ] B0+/F1~F4 Forward/Backward와 파라미터 수를 검토했다.
   - [ ] 공통 분할·증강·시드·학습률 선택 절차, 실행표와 GPU 예산을 동결했다.
4. **실험·분석** (24~28일, C/B/D)  
   - [ ] 최종 조건별 시드 로그, 체크포인트 및 테스트 CSV를 확인했다.
   - [ ] 평균 ± 표본 SD와 학습 곡선, Ablation 결과, F3/F4 분석을 저장했다.
   - [ ] 누락·이탈·시간 제한과 H1~H3 해석을 기록했다.
5. **최종 발표** (29~30일, D/전원)  
   - [ ] 최종 표·그래프, 코드 SHA, 보고서·발표 파일의 수치가 일치한다.
   - [ ] GitHub 새 체크아웃과 README/Colab 단계 및 제출 경로를 검증했다.

## 4. 파일별 구현·검증 작업표

시간은 팀원이 담당 단위를 구현하거나 기존 구현을 인수해 오류를 수정하는 예상 작업 시간이다. GPU 실행은 별도 예산이다. 순서가 같은 행은 병행 가능하다.

| 파일 · 단위 | 담당 | 구현 기능 | 선행 | 순서 | 소요 | 검증 | 완료 기준 |
|---|---|---|---|---|---|---|---|
| `.gitignore`, `data/.gitkeep`, `results/.gitkeep`, `checkpoints/.gitkeep` | A | 큰 파일·비밀값·임시물 제외, 폴더 유지 | Git 루트 확인 | 1 | 20분 | `git check-ignore data/cifar-10-python.tar.gz checkpoints/x/last.pt .env` | 원본 결과/가중치는 Drive, 폴더 표식은 Git |
| `requirements.txt` | A | 공통 Python 패키지, torch 쌍 별도 설치 | 환경 선택 | 1 | 30분 | Mac venv와 Colab import | CUDA torch 유지 |
| `configs/default.yaml` | A | 학습 조건, LR·시드·비율·분할 시드 | 제안서 | 1 | 20분 | `settings.json` 대조 | 요건과 해석된 값 일치 |
| `utils.py` | A | 시드·RNG 복원·YAML·루트·CSV/JSON·원자적 체크포인트 | YAML | 2 | 1시간 | 중단 후 로그/상태 확인 | 체크포인트에서 로그 복원 가능 |
| `dataset.py` 다운로드/증강 | A | CIFAR-10 설치, 학습/검증·테스트 transform | 의존성 | 2 | 45분 | `python dataset.py`, transform 확인 | 검증·테스트 랜덤 증강 없음 |
| `dataset.py` 분할/로더 | A | 클래스 균형 인덱스, 로더 셔플과 seed | YAML·데이터 규격 | 3 | 1시간 | `python -m pytest -q`, 클래스 수와 부분집합 검산 | 45k/5k/10k, 세 비율에서 분할 고정 |
| `models/__init__.py`, `models/resnet.py` | B | 패키지, Stem·3개 Stage, Option A shortcut | 의존성 | 2 | 2시간 | Stage별 shape, B0 gradient | 세 특징 규격과 20층 확인 |
| `models/fusion.py` | B | F1 합, F2 Concat FC, F3 softmax, F4 채널 게이트 | Stage 규격 | 3 | 2시간 | 가중치 합과 gradient | F1~F4 모두 (N,64) 벡터 |
| `model.py` 팩토리/B0·B0+ | B | build_model, Stage 3 GAP, 용량 통제 FC, parameter_count | 백본 | 3 | 1시간 | logits·F2와 파라미터 비교 | (2,10), F2 대비 8개 차이 |
| `model.py` 결합과 Ablation | B | Stage 선택·사영·padding·detach·게이트 반환 | 결합 모듈 | 4 | 1.5시간 | 기본/변형 shape와 backward | 중간층 경로만 detach, F4 게이트 (N,S,64) |
| `tests/test_models.py` | B/A | 6모델·6변형·가중치 및 분할 테스트 | 모델과 분할 | 5 | 1시간 | `python -m pytest -q` | 전부 통과 |
| `train.py` 학습/검증 | A | SGD/Cosine, Epoch 손실·정확도, CPU/CUDA 선택 | 모델과 로더 | 4 | 2시간 | B0 10% 1 Epoch | train/val 값 유한, 테스트 미사용 |
| `train.py` 기록/재개 | A | 설정 스냅샷, best/last, F3 α, 상태·RNG·시간 저장 | 학습 루프 | 5 | 2시간 | Epoch 경계에서 중단 후 동일 옵션으로 재개 | Epoch 중복 없음, 최고 검증 모델과 시간 보존 |
| `evaluate.py` | C | best 모델, 공식 테스트, FLOPs·시간·오차율 CSV 및 F4 게이트 | 학습 체크포인트 | 6 | 1.5시간 | Smoke 체크포인트로 평가, CSV 확인 | 정확도+오차율=100, F4 10×S×64행 |
| `experiments.py` 실행표 | C | 튜닝·핵심 행 생성, ID 하나만 실행/재개 | YAML·train CLI | 6 | 1시간 | tune/core 행 수와 고유 ID 확인 | 54 튜닝행/54 핵심행, 목록 생성만으로 학습 시작 안 함 |
| `experiments.py` LR 선택 | C | 100 Epoch 검증 로그 일치 확인 후 공통 LR 선택 | 튜닝 결과 | 7 | 1시간 | 로그 누락 시 실패, 비율마다 LR 하나 | Seed 0 선택 체크포인트 재사용 |
| `ablation.py` 기준·Stage | C/B | 10%에서 검증 평균 최고 결합과 S3/S2+S3/S1+S3 비교 | 10% 핵심 | 8 | 1시간 | 기준 ID·각 변형 3시드 확인 | 세 변형과 본 실험 S1+S2+S3 연결 |
| `ablation.py` 기타 변형 | C/B | F3 균등·대상 모델 padding/detach | 기준 선정과 옵션 | 8 | 1시간 | argv·reference·18행 확인 | 한 비교당 한 요소만 변경 |
| `analyze.py` 집계/핵심 그림 | C/D | 계획표로 결과 필터링, 비율별 평균·SD·시간·FLOPs, 미완료 ID 표기 | 최종 CSV | 9 | 1.5시간 | n·시드·합계·SD 표본 검산 | 원본 수치 추적, 누락 표시 |
| `analyze.py` 곡선/변형 | C/D | train/val 손실·α·Ablation 짝 차이 그림, F4 클래스·Stage 히트맵 | 로그와 기준 ID | 9 | 1.5시간 | 에포크 축과 시드 짝 대조 | 실제 실행으로만 표/그래프 생성 |
| `colab.ipynb` 환경 | A | CUDA 진단, Drive·GitHub/ZIP, 설치·경로 | 의존성/원격 | 3 | 1시간 | 새 GPU 런타임에서 위부터 실행 | /content/Deep_learning과 Drive 폴더 사용 |
| `colab.ipynb` 실행 | A/C | 다운로드·B0 shape/1 Epoch·정식 ID 실행·최종 평가 | 로더·train·실험표 | 6 | 1시간 | 런타임 초기화 후 Drive 재개 | CSV·last.pt가 Drive에 남음 |
| `README.md`, `PLAN.md` | D/A | 설치·실행·업로드·일정·실험·제출, 저장소와 문서 맞춤 | 전 파일 | 10 | 2시간 | 새 클론에서 안내 실행 | 스크립트와 모든 경로 일치 |

## 5. GitHub 협업 절차

A는 GitHub 저장소 Settings의 브랜치 규칙에서 main 직접 푸시와 강제 푸시를 제한하고, PR과 리뷰 1명 이상을 요구한다. 테스트를 수동으로 수행해도 PR에 명령·결과를 남긴다. 논문 결과용 실행은 코드 변경이 합쳐진 SHA를 기준으로 한다. CSV 실행표의 단일 사본은 C가 관리하고 담당자에게 행을 배정한다.

```bash
git status --short
git switch main
git pull --ff-only origin main
git switch -c feat/b-fusion
# 기능 단위로 수정 후 확인
git diff -- model.py models/fusion.py
python -m pytest -q
git add model.py models/fusion.py tests/test_models.py
git commit -m "Implement latent fusion classifiers"
git push -u origin feat/b-fusion
```

개인 브랜치 예시는 A `feat/a-training`, B `feat/b-fusion`, C `feat/c-experiments`, D `docs/d-report`이다. 기존 브랜치는 새로 생성하는 대신 `git switch <브랜치>`로 이동한다. GitHub에서 base=main과 compare=개인 브랜치를 확인하고 PR에 변경 파일, 시험 방법, 로그 경로를 적는다. 리뷰어는 Files changed에서 파일별 diff를 보고 검증한다. 승인 후 머지하고 다음 작업 전 main을 다시 당긴다.

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
| `core_B0_r100_s0_lrLR100` | B0 | 100% | 0 | LR100 | 100 | A | 선택 후 재사용 예정 | 20~30분 | — | `results/core_B0_r100_s0_lrLR100/` |
| `core_B0_r100_s1_lrLR100` | B0 | 100% | 1 | LR100 | 100 | A | 대기 | 20~30분 | — | `results/core_B0_r100_s1_lrLR100/` |
| `core_B0_r100_s2_lrLR100` | B0 | 100% | 2 | LR100 | 100 | A | 대기 | 20~30분 | — | `results/core_B0_r100_s2_lrLR100/` |
| `core_B0+_r100_s0_lrLR100` | B0+ | 100% | 0 | LR100 | 100 | B | 선택 후 재사용 예정 | 20~30분 | — | `results/core_B0+_r100_s0_lrLR100/` |
| `core_B0+_r100_s1_lrLR100` | B0+ | 100% | 1 | LR100 | 100 | B | 대기 | 20~30분 | — | `results/core_B0+_r100_s1_lrLR100/` |
| `core_B0+_r100_s2_lrLR100` | B0+ | 100% | 2 | LR100 | 100 | B | 대기 | 20~30분 | — | `results/core_B0+_r100_s2_lrLR100/` |
| `core_F1_r100_s0_lrLR100` | F1 | 100% | 0 | LR100 | 100 | B | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F1_r100_s0_lrLR100/` |
| `core_F1_r100_s1_lrLR100` | F1 | 100% | 1 | LR100 | 100 | B | 대기 | 20~30분 | — | `results/core_F1_r100_s1_lrLR100/` |
| `core_F1_r100_s2_lrLR100` | F1 | 100% | 2 | LR100 | 100 | B | 대기 | 20~30분 | — | `results/core_F1_r100_s2_lrLR100/` |
| `core_F2_r100_s0_lrLR100` | F2 | 100% | 0 | LR100 | 100 | C | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F2_r100_s0_lrLR100/` |
| `core_F2_r100_s1_lrLR100` | F2 | 100% | 1 | LR100 | 100 | C | 대기 | 20~30분 | — | `results/core_F2_r100_s1_lrLR100/` |
| `core_F2_r100_s2_lrLR100` | F2 | 100% | 2 | LR100 | 100 | C | 대기 | 20~30분 | — | `results/core_F2_r100_s2_lrLR100/` |
| `core_F3_r100_s0_lrLR100` | F3 | 100% | 0 | LR100 | 100 | C | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F3_r100_s0_lrLR100/` |
| `core_F3_r100_s1_lrLR100` | F3 | 100% | 1 | LR100 | 100 | C | 대기 | 20~30분 | — | `results/core_F3_r100_s1_lrLR100/` |
| `core_F3_r100_s2_lrLR100` | F3 | 100% | 2 | LR100 | 100 | C | 대기 | 20~30분 | — | `results/core_F3_r100_s2_lrLR100/` |
| `core_F4_r100_s0_lrLR100` | F4 | 100% | 0 | LR100 | 100 | C | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F4_r100_s0_lrLR100/` |
| `core_F4_r100_s1_lrLR100` | F4 | 100% | 1 | LR100 | 100 | C | 대기 | 20~30분 | — | `results/core_F4_r100_s1_lrLR100/` |
| `core_F4_r100_s2_lrLR100` | F4 | 100% | 2 | LR100 | 100 | C | 대기 | 20~30분 | — | `results/core_F4_r100_s2_lrLR100/` |
| `core_B0_r25_s0_lrLR25` | B0 | 25% | 0 | LR25 | 100 | A | 선택 후 재사용 예정 | 20~30분 | — | `results/core_B0_r25_s0_lrLR25/` |
| `core_B0_r25_s1_lrLR25` | B0 | 25% | 1 | LR25 | 100 | A | 대기 | 20~30분 | — | `results/core_B0_r25_s1_lrLR25/` |
| `core_B0_r25_s2_lrLR25` | B0 | 25% | 2 | LR25 | 100 | A | 대기 | 20~30분 | — | `results/core_B0_r25_s2_lrLR25/` |
| `core_B0+_r25_s0_lrLR25` | B0+ | 25% | 0 | LR25 | 100 | B | 선택 후 재사용 예정 | 20~30분 | — | `results/core_B0+_r25_s0_lrLR25/` |
| `core_B0+_r25_s1_lrLR25` | B0+ | 25% | 1 | LR25 | 100 | B | 대기 | 20~30분 | — | `results/core_B0+_r25_s1_lrLR25/` |
| `core_B0+_r25_s2_lrLR25` | B0+ | 25% | 2 | LR25 | 100 | B | 대기 | 20~30분 | — | `results/core_B0+_r25_s2_lrLR25/` |
| `core_F1_r25_s0_lrLR25` | F1 | 25% | 0 | LR25 | 100 | B | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F1_r25_s0_lrLR25/` |
| `core_F1_r25_s1_lrLR25` | F1 | 25% | 1 | LR25 | 100 | B | 대기 | 20~30분 | — | `results/core_F1_r25_s1_lrLR25/` |
| `core_F1_r25_s2_lrLR25` | F1 | 25% | 2 | LR25 | 100 | B | 대기 | 20~30분 | — | `results/core_F1_r25_s2_lrLR25/` |
| `core_F2_r25_s0_lrLR25` | F2 | 25% | 0 | LR25 | 100 | C | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F2_r25_s0_lrLR25/` |
| `core_F2_r25_s1_lrLR25` | F2 | 25% | 1 | LR25 | 100 | C | 대기 | 20~30분 | — | `results/core_F2_r25_s1_lrLR25/` |
| `core_F2_r25_s2_lrLR25` | F2 | 25% | 2 | LR25 | 100 | C | 대기 | 20~30분 | — | `results/core_F2_r25_s2_lrLR25/` |
| `core_F3_r25_s0_lrLR25` | F3 | 25% | 0 | LR25 | 100 | C | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F3_r25_s0_lrLR25/` |
| `core_F3_r25_s1_lrLR25` | F3 | 25% | 1 | LR25 | 100 | C | 대기 | 20~30분 | — | `results/core_F3_r25_s1_lrLR25/` |
| `core_F3_r25_s2_lrLR25` | F3 | 25% | 2 | LR25 | 100 | C | 대기 | 20~30분 | — | `results/core_F3_r25_s2_lrLR25/` |
| `core_F4_r25_s0_lrLR25` | F4 | 25% | 0 | LR25 | 100 | C | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F4_r25_s0_lrLR25/` |
| `core_F4_r25_s1_lrLR25` | F4 | 25% | 1 | LR25 | 100 | C | 대기 | 20~30분 | — | `results/core_F4_r25_s1_lrLR25/` |
| `core_F4_r25_s2_lrLR25` | F4 | 25% | 2 | LR25 | 100 | C | 대기 | 20~30분 | — | `results/core_F4_r25_s2_lrLR25/` |
| `core_B0_r10_s0_lrLR10` | B0 | 10% | 0 | LR10 | 100 | A | 선택 후 재사용 예정 | 20~30분 | — | `results/core_B0_r10_s0_lrLR10/` |
| `core_B0_r10_s1_lrLR10` | B0 | 10% | 1 | LR10 | 100 | A | 대기 | 20~30분 | — | `results/core_B0_r10_s1_lrLR10/` |
| `core_B0_r10_s2_lrLR10` | B0 | 10% | 2 | LR10 | 100 | A | 대기 | 20~30분 | — | `results/core_B0_r10_s2_lrLR10/` |
| `core_B0+_r10_s0_lrLR10` | B0+ | 10% | 0 | LR10 | 100 | B | 선택 후 재사용 예정 | 20~30분 | — | `results/core_B0+_r10_s0_lrLR10/` |
| `core_B0+_r10_s1_lrLR10` | B0+ | 10% | 1 | LR10 | 100 | B | 대기 | 20~30분 | — | `results/core_B0+_r10_s1_lrLR10/` |
| `core_B0+_r10_s2_lrLR10` | B0+ | 10% | 2 | LR10 | 100 | B | 대기 | 20~30분 | — | `results/core_B0+_r10_s2_lrLR10/` |
| `core_F1_r10_s0_lrLR10` | F1 | 10% | 0 | LR10 | 100 | B | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F1_r10_s0_lrLR10/` |
| `core_F1_r10_s1_lrLR10` | F1 | 10% | 1 | LR10 | 100 | B | 대기 | 20~30분 | — | `results/core_F1_r10_s1_lrLR10/` |
| `core_F1_r10_s2_lrLR10` | F1 | 10% | 2 | LR10 | 100 | B | 대기 | 20~30분 | — | `results/core_F1_r10_s2_lrLR10/` |
| `core_F2_r10_s0_lrLR10` | F2 | 10% | 0 | LR10 | 100 | C | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F2_r10_s0_lrLR10/` |
| `core_F2_r10_s1_lrLR10` | F2 | 10% | 1 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/core_F2_r10_s1_lrLR10/` |
| `core_F2_r10_s2_lrLR10` | F2 | 10% | 2 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/core_F2_r10_s2_lrLR10/` |
| `core_F3_r10_s0_lrLR10` | F3 | 10% | 0 | LR10 | 100 | C | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F3_r10_s0_lrLR10/` |
| `core_F3_r10_s1_lrLR10` | F3 | 10% | 1 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/core_F3_r10_s1_lrLR10/` |
| `core_F3_r10_s2_lrLR10` | F3 | 10% | 2 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/core_F3_r10_s2_lrLR10/` |
| `core_F4_r10_s0_lrLR10` | F4 | 10% | 0 | LR10 | 100 | C | 선택 후 재사용 예정 | 20~30분 | — | `results/core_F4_r10_s0_lrLR10/` |
| `core_F4_r10_s1_lrLR10` | F4 | 10% | 1 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/core_F4_r10_s1_lrLR10/` |
| `core_F4_r10_s2_lrLR10` | F4 | 10% | 2 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/core_F4_r10_s2_lrLR10/` |

### Ablation 실행표 · 18건

`W`는 10%에서 시드 3개의 최고 검증 정확도 평균이 가장 높은 결합 모델이며 동률이면 F1→F2→F3→F4 순서를 쓴다. `LR10`은 10% 공통 학습률이다. 기준은 같은 시드의 원본 핵심 실행이다. S3 단독은 동일 결합 계열의 Stage 선택 통제를 위해 별도 실행하며, B0 수치는 참고로 병기한다.

| 실험 ID | 모델/변경 | 데이터 비율 | Seed | LR | Epoch | 담당 | 상태 | 예상 시간 | 실제 시간 | 결과 저장 경로 |
|---|---|---:|---:|---|---:|---|---|---|---|---|
| `abl_s3_W_r10_s0_lrLR10` | W · S3만 | 10% | 0 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_s3_W_r10_s0_lrLR10/` |
| `abl_s3_W_r10_s1_lrLR10` | W · S3만 | 10% | 1 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_s3_W_r10_s1_lrLR10/` |
| `abl_s3_W_r10_s2_lrLR10` | W · S3만 | 10% | 2 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_s3_W_r10_s2_lrLR10/` |
| `abl_s23_W_r10_s0_lrLR10` | W · S2+S3 | 10% | 0 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_s23_W_r10_s0_lrLR10/` |
| `abl_s23_W_r10_s1_lrLR10` | W · S2+S3 | 10% | 1 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_s23_W_r10_s1_lrLR10/` |
| `abl_s23_W_r10_s2_lrLR10` | W · S2+S3 | 10% | 2 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_s23_W_r10_s2_lrLR10/` |
| `abl_s13_W_r10_s0_lrLR10` | W · S1+S3 | 10% | 0 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_s13_W_r10_s0_lrLR10/` |
| `abl_s13_W_r10_s1_lrLR10` | W · S1+S3 | 10% | 1 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_s13_W_r10_s1_lrLR10/` |
| `abl_s13_W_r10_s2_lrLR10` | W · S1+S3 | 10% | 2 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_s13_W_r10_s2_lrLR10/` |
| `abl_uniform_F3_r10_s0_lrLR10` | F3 · 균등 α | 10% | 0 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_uniform_F3_r10_s0_lrLR10/` |
| `abl_uniform_F3_r10_s1_lrLR10` | F3 · 균등 α | 10% | 1 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_uniform_F3_r10_s1_lrLR10/` |
| `abl_uniform_F3_r10_s2_lrLR10` | F3 · 균등 α | 10% | 2 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_uniform_F3_r10_s2_lrLR10/` |
| `abl_pad_W_r10_s0_lrLR10` | W · 채널 패딩 | 10% | 0 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_pad_W_r10_s0_lrLR10/` |
| `abl_pad_W_r10_s1_lrLR10` | W · 채널 패딩 | 10% | 1 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_pad_W_r10_s1_lrLR10/` |
| `abl_pad_W_r10_s2_lrLR10` | W · 채널 패딩 | 10% | 2 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_pad_W_r10_s2_lrLR10/` |
| `abl_detach_W_r10_s0_lrLR10` | W · 중간층 detach | 10% | 0 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_detach_W_r10_s0_lrLR10/` |
| `abl_detach_W_r10_s1_lrLR10` | W · 중간층 detach | 10% | 1 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_detach_W_r10_s1_lrLR10/` |
| `abl_detach_W_r10_s2_lrLR10` | W · 중간층 detach | 10% | 2 | LR10 | 100 | C | 대기 | 20~30분 | — | `results/abl_detach_W_r10_s2_lrLR10/` |

## 7. 우선순위와 범위 조정

| 순위 | 지킬 작업 | 일정 판단 |
|---|---|---|
| P0 | CUDA 연결, CIFAR-10 분할, B0, 학습·평가, 시드·로그, Drive 재개 | 정식 GPU 큐 전에 모두 검증 |
| P1 | B0+, F1~F4, 세 시드·동일 튜닝·공통 LR, 결과표와 학습 곡선 | 방법론 30%, 해석 25%를 고려해 완전한 비교 묶음을 우선 확보 |
| P2 | 4종 Ablation, F3/F4 내부 가중치, Linear Probe, Welch 참고 검정, 혼동 행렬과 추가 그림 | 분석 배점에 직접 연결되는 detach·Stage, 균등·패딩 및 기본 α/게이트부터 수행. Probe·검정·추가 그림은 후순위 |

23일 저녁과 25일 종료 시 C가 `남은 학습 수 × 비율별 실측`을 슬롯별로 계산한다. 모자라면 25% 전체를 먼저 제외하고 100%·10% 비교의 3시드와 공통 조건을 보존한다. 25% 일부를 이미 실행했다면 보조 로그에 남겨 최종 동일 조건 표에서 분리한다. 더 부족하면 100% 완전한 묶음 확보 및 10% 가능한 범위와 한계 보고를 우선하고 H3 판정을 보류한다. Ablation도 가능한 변형의 세 시드를 완결한다. 변경 날짜, 결정자, 이유, 예상/실측 시간, 영향 받은 실험 ID와 가설을 실행표 및 보고서에 적는다. 에폭·배치·시드·데이터 분할을 일부 모델만 바꾸거나 누락 값을 만들어 채우지 않는다.

## 8. 예상 문제와 해결 절차

| 문제 | 가능한 원인 | 확인 | 해결 절차 (담당) |
|---|---|---|---|
| Colab GPU 할당 실패 | GPU 할당량 소진 또는 CPU 런타임 | 런타임 유형, `nvidia-smi`, `torch.cuda.is_available()` | GPU 선택 후 재연결 → 팀원 가용 슬롯으로 ID 인수 → 동일 설정 재실행 (A) |
| 런타임 중단 | 유휴·시간 한도, 연결 해제 | Drive 마지막 `last.pt`, history Epoch | 새 런타임에서 동일 커밋/환경·Drive 연결 → 같은 ID `--resume` → 중복 Epoch 확인 (A) |
| GPU 메모리 부족 | 다른 GPU 프로세스, 메모리 단편화, 배치 용량 | `nvidia-smi`, CUDA OOM 위치 | 사용 안 하는 텐서/프로세스 정리 → 새 GPU로 재개 → 반복되면 전 비교군 공통 프로토콜 변경을 검토하고 기존 설정과 구분 (A/B) |
| CIFAR-10 다운로드 오류 | 서버·DNS·불완전 아카이브 | traceback, 다운로드 크기, torchvision 무결성 메시지 | 연결 재시도 → 정상 아카이브를 다른 팀 환경에서 확보 → 새 데이터 디렉터리에 복사 후 다운로드/무결성 재확인 (A) |
| PyTorch/CUDA 버전 충돌 | torchvision과 torch 불일치, CUDA 패키지 덮어쓰기 | 버전 출력, `nvidia-smi`, import traceback | 깨끗한 GPU 런타임에서 기본 CUDA 쌍 사용 → requirements만 설치 → Smoke (A) |
| 차원 오류 | Stage 순서, GAP 축, Concat 폭 | shape 로그, `pytest` | 표준 shape 16/32/64 확인 → 64차원 사영 후 결합 → logits (N,10) 및 backward 확인 (B) |
| GitHub 머지 충돌 | 같은 줄 또는 공유 인터페이스 변경 | `git status`, `git diff --name-only --diff-filter=U` | 공유한 호출 규격과 양쪽 변경 확인 → 충돌 표지 제거·테스트 → 파일 add·commit·push (파일 담당) |
| 체크포인트 저장 실패 | Drive 미연결, 용량/권한 부족 | 쓰기 테스트, Drive 용량, `last.pt` 읽기 | 재마운트 → 경로/여유 확인 → 마지막 정상 파일에서 재개 → CSV Epoch 대조 (A) |
| 학습 과다 소요 | 100% 로더, 느린 GPU, Drive 쓰기 | 비율별 Epoch 중앙값, GPU 사용률, 저장 시간 | 병목 기록 → 이용 가능 슬롯 병렬 배치 → 25% 전체 축소 및 결정 기록 (A/C) |
| 재현 실패 | 커밋·버전·split seed·옵션 차이 | `settings.json`, SHA, 분할/시드, GPU 종류·로그 | 고정 커밋과 환경 재구성 → 같은 명령을 새 진단 ID로 실행 → 오차 원인과 동일 환경 재현 범위 기록 (A/C) |
| 성능 개선 미확인 | 시드 변동, Residual 정보 보존, 파라미터 증가 등 | 두 설정의 표본 SD, B0+, 손실 곡선, Ablation | 차이와 SD 및 방향을 수치로 보고 → α/게이트·detach 결과로 가능한 설명 검토 → 추정 설명을 한계로 표시 (C/D) |

## 9. 분석 순서와 최종 제출 확인표

C는 `summary.csv`와 `ablation_summary.csv`의 n을 확인하고 동일 비율·LR에서 시드별 오차율을 원본과 맞춘다. Ablation 차이는 변형 오차율 − 같은 시드 기준 오차율이며 양수는 변형에서 악화된 경우다. `*_loss.png`로 train/val 격차와 수렴을 비교한다. F3는 `*_alpha.png`와 검증 최고 시점의 α를 확인한다. F4는 `*_gates.png`와 `gates.csv`를 확인해 클래스별 채널 평균을 비교하고 시드 간 방향이 일치하는지 검토한다. 게이트 크기와 분류 개선의 인과 관계는 Ablation 근거와 함께 해석한다.

| 확인 | 제출 대상 | 담당 | 최종 확인 |
|---|---|---|---|
| [ ] | 제안서 | D | Track A·질문·H1~H3·CIFAR-10/ResNet-20 일치 |
| [ ] | 정식 실험 계획/실행표 | C | 54/18 및 튜닝 순증, 또는 축소 사유·누락 ID·학습 시간 표시 |
| [ ] | 보고서 4~6쪽 | D | 연구 질문, 통제, 공개 B0 비교, 표·그래프·해석, 오차율·SD·효율, 한계 |
| [ ] | 발표 자료 | D/B | 모델 도식, 핵심 그림, Ablation, 결론·한계, 리허설 |
| [ ] | GitHub 최종 코드 및 Colab | A/B | 고정 SHA, 새 클론, README 명령, CUDA Smoke와 Drive 재개 |
| [ ] | 재현 자료 | A/C | YAML, 시드·실행 ID·CSV·체크포인트 경로, Drive 읽기 권한, 결과 출처 |
| [ ] | 제출 준비 | 전원 | 공유 대상과 제출 위치 확인, 표/그래프/발표 수치가 같은 커밋과 결과를 가리킨다 |

### 계획과 저장소 대조

파일별 표의 모든 코드·문서·폴더는 저장소에서 확인한다. `experiments.py`는 전체 학습을 자동 실행하지 않으며 각 ID를 팀이 배정한다. 기본 분석은 최종 CSV·loss·α·F4 게이트 히트맵 및 Ablation 차이를 생성하고 F4 집계값은 `evaluate.py`가 기록한다. Linear Probe, Welch 검정, 혼동 행렬, 보고서와 발표 자료는 후속 일정에 배치했다.

### 스캐폴드 검증 기록

- Mac CPU에서 모델·Ablation·데이터 분할 자동 테스트 15개를 통과했다.
- 공식 CIFAR-10, 10%, 시드 0으로 B0 1 Epoch 학습 및 독립 테스트 평가를 완료했다. 학습·검증 구간은 약 20.8초였다. 이 실행은 동작 확인용이며 정식 비교표에 포함하지 않는다.
- 합성 입력으로 Epoch 중단 후 재개, 평가·FLOPs 저장, F3 α 및 F4 게이트 수집과 분석 출력을 확인했다. 검증 로그 픽스처로 공통 LR 선택·시드 0 재사용·실험표 건수를 확인했다.
- Colab 코드 셀의 Python 문법을 검증했다. 실제 Colab CUDA 및 Drive 연결·재개와 정식 학습은 팀 일정에서 확인한다.
