# Deep_learning
심층학습 기말 프로젝트

CIFAR-10에서 ResNet-20의 중간층과 마지막 층 표현을 여러 방식으로 결합하고, 동일 조건에서 분류 오차율을 비교합니다. 일정, 역할, 실행 순서 및 실패 대응은 [PLAN.md](PLAN.md)에 있습니다.

## Mac과 VS Code

Git 저장소가 들어 있는 폴더를 VS Code에서 열고 터미널에서 다음을 실행합니다. VS Code 인터프리터는 `.venv/bin/python`을 선택합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -m pytest -q
python dataset.py
python train.py --model B0 --ratio 10 --seed 0 --epochs 1 --run-id smoke_b0
python evaluate.py checkpoints/smoke_b0/last.pt --flops
```

기본 경로는 저장소 안의 `data/`, `results/`, `checkpoints/`입니다. Smoke 결과는 최종 비교에 넣지 않습니다. `last.pt`에는 마지막 에포크와 최고 검증 정확도 시점의 모델이 함께 들어 있습니다. 중단하면 동일 명령에 `--resume`을 추가하며, 원래 에포크 목표와 실행 조건을 유지합니다. 같은 실행 ID를 두 런타임에서 동시에 사용하지 않습니다. 체크포인트를 로드할 때는 팀에서 관리한 파일을 사용합니다.

## GitHub와 Colab

기존 `origin`을 `git remote -v`로 확인합니다. 현재 변경 사항을 팀 브랜치에 올리고 GitHub에서 PR을 생성합니다.

```bash
git switch -c chore/track-a-scaffold
git status --short
git add README.md PLAN.md .gitignore colab.ipynb requirements.txt configs models tests data/.gitkeep results/.gitkeep checkpoints/.gitkeep *.py
git diff --cached --stat
git commit -m "Add Track A project scaffold"
git push -u origin chore/track-a-scaffold
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
