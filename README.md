# Deep_learning
심층학습 기말 프로젝트

CIFAR-10에서 ResNet-20의 중간층과 마지막 층 표현을 여러 방식으로 결합하고, 동일 조건에서 분류 오차율을 비교합니다. 일정, 역할, 실행 순서 및 실패 대응은 [PLAN.md](PLAN.md)에 있습니다.

## Mac과 VS Code

Git 저장소가 들어 있는 폴더를 VS Code에서 열고 터미널에서 다음을 실행합니다. VS Code 인터프리터는 `.venv/bin/python`을 선택합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -c "import torch, torchvision, yaml, matplotlib, pytest, fvcore; print(torch.__version__)"
python -m pytest -q
python dataset.py
python train.py --model B0 --ratio 10 --seed 0 --epochs 1 --run-id smoke_b0
python evaluate.py checkpoints/smoke_b0/last.pt --flops
```

`dataset.py`는 CIFAR-10을 처음 실행할 때 다운로드하므로 인터넷 연결이 필요합니다. 기본 경로는 저장소 안의 `data/`, `results/`, `checkpoints/`입니다. Smoke 결과는 최종 비교에 넣지 않습니다. `last.pt`에는 마지막 에포크와 최고 검증 정확도 시점의 모델이 함께 들어 있고 `best.pt`는 최고 검증 정확도를 기록한 에포크의 상태입니다. 중단하면 동일 명령에 `--resume`을 추가하며, 원래 에포크 목표와 실행 조건을 유지합니다. 같은 실행 ID를 두 런타임에서 동시에 사용하지 않습니다. 체크포인트를 로드할 때는 팀에서 관리한 파일을 사용합니다.

기본 DataLoader 작업자 수는 Mac에서도 바로 실행되도록 0입니다. Colab에서 작업자를 늘리려면 `--workers 2`를 지정합니다. 자동 장치 선택은 CUDA → Apple MPS → CPU 순서이며, MPS 연산 오류가 나면 `--device cpu`와 새 실행 ID로 다시 실행합니다.

## 파일 구성

| 경로 | 용도 |
|---|---|
| `configs/default.yaml` | 데이터 분할, 배치 크기, 학습 기본 설정 |
| `dataset.py` | CIFAR-10 다운로드, 분할, 데이터 로더 |
| `model.py`, `models/` | ResNet-20과 분류 모델 |
| `train.py`, `utils.py` | 학습, 기록, 체크포인트와 재개 |
| `evaluate.py`, `experiments.py`, `analyze.py` | 최종 평가, 실행표, 분석 |
| `data/` | 원본 CIFAR-10, Git 업로드 제외 |
| `results/`, `checkpoints/` | CSV 결과와 모델 상태, Git 업로드 제외 |

PyTorch와 torchvision은 Mac에서는 위 순서대로 설치하고, Colab에서는 GPU 런타임에 기본 설치된 CUDA 호환 조합을 먼저 확인합니다. Colab에서는 다음 명령으로 CUDA가 보이는지 확인한 뒤 나머지 패키지만 설치합니다.

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m pip install -r requirements.txt
```

CUDA가 보이지 않으면 Colab의 GPU 런타임 설정과 PyTorch 설치 상태를 확인한 뒤 학습을 시작합니다. `requirements.txt`에는 사용 중인 패키지가 모두 있지만, PyTorch 쌍은 플랫폼에 맞는 설치본을 먼저 준비합니다. 설치된 버전이 요구 범위를 충족하면 `pip install -r requirements.txt`는 해당 쌍을 다시 설치하지 않습니다.

## GitHub와 Colab

기존 `origin`을 `git remote -v`로 확인합니다. 작업 브랜치에서 검증한 뒤 GitHub의 main 대상 PR을 만듭니다. main 브랜치 보호 설정은 저장소 관리자가 GitHub의 Settings → Rules → Rulesets에서 대상 브랜치를 `main`으로 지정하고 PR 필수, 최소 승인 1명, 강제 푸시 차단을 설정합니다.

```bash
git status --short
git add .gitignore requirements.txt README.md configs/default.yaml dataset.py train.py utils.py tests data/.gitkeep results/.gitkeep checkpoints/.gitkeep
git diff --cached --stat
git commit -m "Complete DS-01 through DS-03"
git push -u origin HEAD
```

Colab에서 `colab.ipynb`를 열고 런타임 유형을 GPU로 바꿉니다. 저장소 URL과 팀이 확정한 커밋을 설정한 뒤 위에서부터 실행합니다. CUDA용 torch/torchvision은 Colab에 설치된 조합을 유지합니다. 팀 저장소가 비공개이면 인증된 환경에서 팀 커밋을 ZIP으로 내보내 Drive에 보관하고 노트북의 ZIP 옵션을 씁니다. 정식 학습 셀은 실험표와 실행 ID를 지정했을 때 한 건을 실행합니다.

## 정식 실험 CLI

```bash
python experiments.py tune
python experiments.py run --output results/tune_plan.csv --run-id core_B0_r100_s0_lr0.05
# 동일 방식으로 tune_plan의 행을 실행한 뒤 다음 명령을 실행합니다.
python experiments.py select
python experiments.py core
# core_plan의 재사용 대상은 건너뛰고 시드 1, 2를 실행합니다.
python experiments.py run --output results/core_plan.csv --run-id <실행ID>
python ablation.py
python experiments.py run --output results/ablation_plan.csv --run-id <실행ID>
python evaluate.py checkpoints/<실행ID>/last.pt --flops
python analyze.py results/core_plan.csv results/ablation_plan.csv
```

`<실행ID>` 부분은 실제 CSV 값으로 교체합니다. 테스트 평가는 비교 조건과 Ablation 대상을 검증 값으로 고정한 뒤 수행합니다. FLOPs는 fvcore 기준이며 FMA 한 번을 한 FLOP으로 집계하고 미지원 연산을 출력합니다. F3 α는 `history.csv`, F4 클래스·Stage·채널별 평균 게이트는 `gates.csv`에 저장합니다. 팀 제출용 요약과 실험표는 Drive의 원본 경로 및 커밋과 함께 공유합니다.
