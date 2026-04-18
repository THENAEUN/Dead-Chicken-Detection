# Dead-Chicken-Detection & Behavioral Analysis

---

## 프로젝트 개요
본 프로젝트는 스마트 양계 환경을 위한 **AI 기반 폐사체 조기 탐지 솔루션**입니다. YOLOv8 객체 탐지 모델을 통해 닭의 위치를 실시간으로 추적하고, 개별 ID에 부여된 이동 궤적을 분석하여 폐사 의심 개체를 자동 판별합니다.

## 주요 기능
* **그리드 기반 처리 (Grid-based Processing):** 고해상도 프레임을 $3 \times 3$ 그리드로 분할하여 탐지 효율을 최적화합니다.
* **상태 기반 모니터링 (State-based Monitoring):** 각 객체를 행동에 따라 세 가지 상태로 분류합니다.
  * **Active (초록색):** 정상적인 움직임이 관찰되는 상태.
  * **Stationary (노란색):** 150프레임 이상 이동이 없는 상태 (휴식 중).
  * **Dead (빨간색):** 450프레임 이상 이동이 없어 폐사가 의심되는 상태.
* **상태 복구 로직 (State Recovery Logic):** 객체의 움직임이 감지되는 즉시 상태를 'Active'로 복구하여 오탐(False Positive)을 최소화합니다.
* **데이터 로깅 (Data Logging):** 프레임별 개체 상태 데이터를 `chicken_status_log.csv` 파일로 자동 내보냅니다.

## 방법론
시스템은 연속된 프레임 사이에서 객체 중심점(Centroid) 간의 **유클리드 거리 ($d$)**를 계산하여 활동성을 평가합니다.

$$d = \sqrt{(x_{t+1}-x_t)^2 + (y_{t+1}-y_t)^2}$$

계산된 거리 $d$가 특정 임계값 $\epsilon$보다 낮은 상태로 일정 프레임 이상 유지될 경우, 시스템은 경고 또는 폐사 알람을 발생시킵니다.

## 한계점 및 향후 과제
1. **오탐 개선 (False Positives):** 단순 수면 상태와 폐사를 정교하게 구분하기 위해 **Optical Flow** 알고리즘을 도입, 미세한 호흡 움직임을 감지할 예정입니다.
2. **폐착 문제 (Occlusion):** 객체 겹침 시 ID 유지력을 높이기 위해 **Re-identification(Re-ID)** 로직을 보완할 계획입니다.
3. **동적 임계값 (Dynamic Thresholds):** 일주기 리듬(Circadian Rhythm)에 따른 활동량 차이를 반영하여 가변 임계값 시스템을 구축할 예정입니다.

## 📂 저장소 구조
```text
├── src/
│   ├── tracker.py             # YOLOv8 기반 객체 추적 로직
│   └── movement_analysis.py   # 거리 계산 및 상태 전이 엔진
├── data/                      # 가중치 파일 및 샘플 영상 (Private)
└── results/                   # 분석 결과 로그 및 데모 이미지
