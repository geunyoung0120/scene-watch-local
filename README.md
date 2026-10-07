# Scene Watch — 폭행 감지 로컬 웹앱

노트북 카메라의 사진을 **DINOv2 CLS → 로지스틱 회귀 → 최근 5장 확률 평균**으로 분석합니다. 브라우저에는 실시간 카메라 영상과 폭행 분류 점수를 나란히 표시합니다.

**로컬 주소: http://127.0.0.1:8876** — 설치·학습 후 서버를 실행해야 접속할 수 있습니다. GitHub 저장소 공개는 웹 추론 서버 호스팅이 아닙니다. Python/PyTorch 추론이 필요하므로 GitHub Pages만으로 실행할 수 없습니다.

## 현재 모델과 한계

현재 적용 버전은 **people-v3 (2026-10-08)**입니다. 아래 AIRTLab 구성에 육안 검토한 COCO 비폭행 사진 548장(학습 355 / 검증 81 / 시험 112)을 추가했습니다. 합계 원본 1,648장과 변형 4,000장이며, 실제 학습 입력은 5,155장입니다. 새 자료는 군중, 크게 보이는 인물, 일반 인물 장면과 배경 사진입니다.


- DINOv2 Small의 가중치는 고정하고 384차원 CLS 토큰으로 로지스틱 회귀만 학습합니다.
- 실제 촬영 시점은 AIRTLab의 **카메라 두 개**입니다. 원본 사진 1,100장 중 학습 800장, 검증 100장, 시험 200장입니다.
- 학습 사진에 좌우 반전, ±8° 기울기, 두 방향의 작은 원근 변형을 적용해 **변형 사진 4,000장**을 추가했습니다. 구도 보강 단계의 학습 입력은 4,800장이며, 현재는 새 비폭행 학습 사진 355장이 더해졌습니다. 이는 실제 새로운 3D 촬영 시점이 아닙니다.
- 같은 장면의 두 카메라와 파생 사진은 같은 분할에 묶습니다. 검증·시험 원본은 초기 모델과 동일합니다.
- 폭행 학습 자료는 단일 장소와 반복된 배우의 연출 영상이며, 해당 프레임은 영상 라벨을 상속합니다. 폭행 영상의 대기 자세도 폭행 라벨일 수 있습니다. 새로운 사람·장소·노트북 시점의 정확도는 검증하지 않았습니다.
- **자기 몸을 때리는 동작은 학습·검증하지 않았습니다.** 낮은 점수가 해당 동작이 없다는 뜻은 아닙니다. 모델 점수는 실제 폭행 발생 확률이나 안전 보장이 아닙니다.
- 5장 평균은 점수 변동을 줄이며, 움직임 방향이나 시간적 행동 패턴을 학습하지 않습니다.

## 설치 및 초기 모델 학습

검증 환경은 Python 3.13, Apple M4 Pro, macOS입니다. Apple GPU가 있으면 MPS, 그 외에는 CPU를 사용합니다. `.command` 실행 파일과 `manage_server.py --open`은 macOS용입니다. 다른 환경에서는 `scripts/serve.py`로 서버를 실행하고 브라우저에서 로컬 주소를 직접 여세요.

학습 데이터, 이미지 미리보기, 모델 가중치, 임베딩 캐시, 카메라 자료는 GitHub에 포함하지 않습니다. 최초 설치에는 인터넷과 충분한 디스크 공간이 필요합니다. 다음 명령은 공식 출처에서 고정된 모델과 연구·교육용 자료를 내려받습니다.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/train.py
```

아래 비교 전에 **최초 실행에서 한 번만** 초기 모델을 보관합니다. 이미 `baseline-v1`이 있으면 덮어쓰지 마세요.

```sh
mkdir models/baseline-v1
cp models/violence_head.json models/baseline-v1/violence_head.json
cp reports/training.json models/baseline-v1/training.json
cp reports/predictions.csv models/baseline-v1/predictions.csv
```

## 구도 변화 자료 추가 학습

```sh
.venv/bin/python scripts/expand_views.py
.venv/bin/python scripts/train.py --manifest data/views-v2/manifest.csv --output-dir models/views-v2
.venv/bin/python scripts/evaluate_views.py
.venv/bin/python scripts/activate_views.py
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/manage_server.py start --open
```

`expand_views.py`는 미사용 장면 5개씩(폭행·비폭행)을 두 카메라에서 추출해 원본 100장을 추가하고, 학습용 원본 800장에만 다섯 가지 변형을 적용합니다. 사진과 매니페스트는 `data/views-v2/`, 미리보기는 `reports/views-v2/augmentation-preview.jpg`에 생성됩니다.

`evaluate_views.py`는 고정된 `models/baseline-v1/violence_head.json`과 새 모델을 비교합니다. 적용 조건은 **원본+변형 검증 사진의 log loss가 기존보다 나쁘지 않고, 원본 검증 정확도 하락이 5%p 이하**인 것입니다. 시험 지표는 적용 결정에 사용하지 않습니다. 학습 스크립트도 시험 지표를 출력하므로 완전히 봉인한 최종 시험 절차는 아닙니다.

조건을 통과하면 `activate_views.py`가 평가 당시 모델·매니페스트의 해시를 확인하고 새 분류기를 적용합니다. 통과하지 않으면 기존 모델을 유지합니다. 실행 중인 서버에는 재시작 후 반영됩니다.

```sh
.venv/bin/python scripts/manage_server.py stop
.venv/bin/python scripts/manage_server.py start --open
```

매니페스트나 전처리를 바꾸면 해당 학습 출력 폴더의 `embeddings.npz`를 제거하고 다시 추출해야 합니다. 캐시 지문이 다르면 학습은 중단합니다.

## 일상 인물·군중 오탐 보강 (현재 v3)

v2 재현을 마친 뒤 아래 명령을 실행합니다. 첫 실행에서 현재 v2 모델을 `models/baseline-v2/`에 자동 보관하며 기존 백업은 덮어쓰지 않습니다.

```sh
.venv/bin/python scripts/prepare_people.py
.venv/bin/python scripts/train_people.py
.venv/bin/python scripts/activate_people.py
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/manage_server.py stop
.venv/bin/python scripts/manage_server.py start --open
```

COCO 2017 공식 주석의 사진별 라이선스가 CC BY 2.0 또는 CC BY-SA 2.0인 후보 560장을 내려받았습니다. 군중은 사람 주석 5개 이상, 근접 인물은 가장 큰 사람 바운딩 박스 면적이 화면의 20% 이상인 조건입니다. 실제 거리나 얼굴 크기를 측정한 것은 아니며 배경의 작은 사람이나 신체 일부도 포함됩니다. 설명문에서 폭행 가능성이 있는 자료를 먼저 제외하고 14개 연락표의 모든 후보를 확인했습니다. 화면 속 사진·애매한 접촉·넘어짐 등 12장을 제외한 548장만 비폭행으로 사용했습니다. 검토한 이미지 ID·SHA256과 제외 이유는 [검토 기록](docs/people-review.json)에 있습니다. `data/people-v3/candidates.json` 및 매니페스트에는 원본 URL, Flickr 사진 페이지, 라이선스가 보존됩니다.

유사 이미지 dHash 거리 5 이하 및 동일 파일을 같은 그룹으로 묶고 분할합니다. 인물 신원이나 사진가별 독립 분할을 보장하지는 않습니다. AIRTLab 원본과 다섯 변형은 합산 가중치 1로 처리해 파생 사진 수만으로 학습을 지배하지 않도록 했습니다. 정규화 5개 × 추가 비폭행 가중치 3개, 총 15개 설정을 **검증 자료만으로** 비교했습니다. 기존 검증 폭행 재현율 하락 5%p 이내, 기존 비폭행 오탐 상승 2%p 이내, 새 비폭행 오탐 감소를 요구했습니다. 선택된 모델을 저장한 뒤 시험 결과를 계산했습니다. 분류 임계값 0.5와 5장 평균은 바꾸지 않았습니다.

| 시험 항목 | 기존 v2 | 새 v3 |
|---|---:|---:|
| 새 비폭행 사진 전체 112장 오탐 | 70장 (62.5%) | 0장 |
| 군중 33장 오탐 | 25장 | 0장 |
| 큰 인물 사진 39장 오탐 | 20장 | 0장 |
| 기존 AIRTLab 원본 사진 정확도 | 75.0% | 79.0% |
| 기존 영상별 5장 평균 정확도 | 85.0% | 90.0% |
| 기존 영상별 폭행 재현율 | 90.0% | 95.0% |
| 기존 이미지별 폭행 재현율 | 87.0% | 85.0% |

0건은 이 112장 시험에서 관측한 결과이며 실제 오탐 확률이 0이라는 뜻은 아닙니다. 새 사진에는 비폭행만 있으므로 **새 환경의 폭행 재현율은 검증되지 않았습니다.** 모델이 데이터 출처나 배경을 단서로 사용할 가능성이 남아 있습니다. 기존 AIRTLab 시험은 여러 버전에서 사용한 회귀 검사이며 새로운 최종 시험 자료가 아닙니다. 실제 사용자 카메라 사진을 수집하거나 학습하지 않았습니다. 모델 점수는 실제 사고 발생 확률로 보정되지 않았습니다.

[상세 비교·검증 설정](docs/people-evaluation.json)에서 정확도 외의 log loss, 재현율, 오탐률도 확인할 수 있습니다. 일부 지표는 악화됐으며 모든 상황의 개선을 의미하지 않습니다. 원본은 로컬 `data/people-v3/images/`, 검토 연락표는 `reports/people-v3/review-*.jpg`에 있습니다. 자료는 GitHub로 재배포하지 않습니다.

## 사용과 종료

1. macOS에서는 `웹앱 시작.command`를 더블클릭합니다.
2. 브라우저에서 **카메라 시작**을 누르고 카메라 사용을 허용합니다.
3. 왼쪽 미리보기와 오른쪽 최근 5장 평균을 확인합니다. 처음 네 장은 수집 중으로 표시합니다.
4. **중지**를 누르면 카메라를 끄고 평균을 초기화합니다. 다른 탭으로 이동해 페이지가 숨겨져도 중지합니다.
5. 서버까지 끄려면 `웹앱 종료.command`를 실행합니다.

카메라 권한 문제는 브라우저 사이트 권한과 macOS의 개인정보 보호 및 보안 → 카메라에서 확인하세요. 이미 사용 중인 다른 카메라 앱도 확인하세요.

## 이전 구도 보강 결과 (v2)

2026-10-06 실행 결과입니다. 같은 시험 원본 200장·40개 영상·20개 독립 장면에서 비교했습니다. 변형 시험 사진 1,000장은 이 200장을 변형한 것이므로 독립 표본 수가 늘어난 것은 아닙니다.

| 시험 지표 | 초기 모델 | 구도 보강 모델 |
|---|---:|---:|
| 원본 사진 정확도 | 78.0% | 75.0% |
| 변형 사진 정확도 | 74.3% | 76.5% |
| 원본 영상별 5장 평균 정확도 | 85.0% | 85.0% |
| 원본 영상별 폭행 재현율 | 95.0% | 90.0% |
| 원본 영상별 폭행 정밀도 | 79.2% | 81.8% |

변형 사진의 정확도는 개선됐지만 원본 사진과 일부 지표는 하락했습니다. 모든 상황에서 더 좋아진 모델로 해석하면 안 됩니다. 검증 원본+변형 log loss는 약 0.500 → 0.407로 개선돼 미리 정한 적용 기준을 통과했습니다.

영상별 다섯 장은 영상 길이의 20·35·50·65·80%에서 추출합니다. 실시간 0.4초 창의 정확도를 직접 측정한 결과가 아닙니다.

- [모델·카메라·변형별 비교](docs/evaluation.json)
- [학습 및 원본 시험 결과](docs/training.json)
- [데이터 구성](docs/dataset-summary.json)
- [이 노트북의 벤치마크](docs/benchmark.json)

현재 모델의 JPEG 해석부터 5장 평균까지 중앙값 약 **10.2ms**, p95 약 **11.2ms**였습니다. 기본 분석 간격은 실측 p95를 사용한 `max(100, ceil(p95 × 1.25 / 10) × 10)`ms로, 이 노트북에서는 **100ms**입니다. 파일 벤치마크는 카메라 캡처·브라우저 JPEG 생성·HTTP 왕복 시간을 제외합니다.

5장 사이의 시간은 약 400ms이며 영상 미리보기 프레임 속도와 분석 빈도는 별개입니다. 요청은 순차 처리하고 밀린 프레임을 쌓지 않습니다. 5초 이상의 공백 또는 중지·재시작 시 평균을 초기화합니다.

## 로컬 처리와 검증

서버는 `127.0.0.1`에만 바인딩합니다. 카메라 사진은 브라우저에서 같은 노트북의 서버로 전달되고 메모리에서 추론한 뒤 버립니다. 사진·영상·음성 저장이나 외부 전송은 하지 않습니다. 초기 설치 후 추론에는 인터넷이 필요하지 않습니다. Host/Origin/요청 토큰, JPEG 크기, 세션·프레임 순서를 검사합니다.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m playwright install chromium
# 아래 두 검사는 로컬 서버를 켠 상태에서 실행합니다.
.venv/bin/python scripts/verify_start_stop.py
.venv/bin/python scripts/verify_web.py
```

브라우저 검사는 녹화된 AIRTLab 영상을 가상 카메라로 사용하며 물리 카메라를 켜지 않습니다. 사용 중인 분석 세션을 초기화하므로 실제 사용을 마친 뒤 검사를 실행하세요.

## 출처와 라이선스

이 저장소의 직접 작성한 코드는 [MIT License](LICENSE)로 공개합니다. **이 라이선스는 외부 데이터나 모델에 적용되지 않습니다.**

- 데이터: [AIRTLab 제작자 저장소](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos), 연구·교육용 무료. 사용 조건은 공식 원문을 확인하세요. 원본·추출 프레임·변형 이미지·학습된 분류기 등 데이터 관련 산출물은 이 저장소에서 재배포하지 않습니다.
- 추가 비폭행 사진: [COCO 공식 이용조건](https://cocodataset.org/#termsofuse). 주석은 CC BY 4.0, 선택한 사진은 개별 CC BY 2.0/CC BY-SA 2.0입니다. COCO는 사진 저작권을 소유하지 않으므로 개별 조건을 따릅니다. 다운로드·선정·검토 기록을 보존하고 사진과 학습 모델은 재배포하지 않습니다.
- 데이터 버전: `1f7747e104301ccaa82ef5a2f6804b51ced1c398`. 영상 파일의 Git blob 해시를 확인합니다.
- 인용: M. Bianculli et al., *A dataset for automatic violence detection in videos*, Data in Brief 33 (2020), [doi:10.1016/j.dib.2020.106587](https://doi.org/10.1016/j.dib.2020.106587).
- 특징 추출기: [Meta DINOv2 Small](https://huggingface.co/facebook/dinov2-small), Apache-2.0. 고정 버전 `ed25f3a31f01632728cabb09d1542f84ab7b0056`, safetensors 사용, 실행 시 원격 코드 로드 없음.

## 주요 파일

| 경로 | 내용 |
|---|---|
| `web/` | 카메라 미리보기, 점수, 추세 그래프 |
| `violence_app/` | 전처리·추론·평균·로컬 HTTP 서버 |
| `scripts/` | 다운로드, 데이터 생성, 학습, 평가, 적용, 벤치마크 |
| `tests/` | 단위·HTTP 검증 |
| `docs/` | 실제 실행한 수치 보고서 |
| `data/`, `models/`, `reports/` | 로컬에서 생성되는 자료, Git 제외 |

최초 v1 모델로 복구하려면 서버를 멈추고 `models/baseline-v1/`의 `violence_head.json`을 `models/`, `training.json`과 `predictions.csv`를 `reports/`에 복사한 뒤 벤치마크와 서버를 다시 실행하세요. 복구 시 웹 하단의 자료 수 설명은 현재 v3 기준으로 남으므로 필요하면 함께 수정하세요.

현재 모델 직전 버전으로 복구할 때는 `models/baseline-v2/`에 보관한 모델과 보고서를 사용하세요. 위 v1 복구는 최초 버전으로 돌아가는 절차입니다.
