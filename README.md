# 🐔 Dead-Chicken-Detection & Behavioral Analysis

> **An intelligent poultry monitoring system using YOLOv8 and custom behavioral analysis logic to detect mortality in real-time.**

---

## 📌 Project Overview
본 프로젝트는 스마트 양계 환경을 위한 **AI 기반 폐사체 조기 탐지 솔루션**입니다. YOLOv8 객체 탐지 모델을 통해 닭의 위치를 실시간으로 추적하고, 개별 ID에 부여된 이동 궤적을 분석하여 폐사 의심 개체를 자동 판별합니다.

## 🚀 Key Features
* **Grid-based Processing:** Optimizes detection by dividing high-resolution frames into $3 \times 3$ grids.
* **State-based Monitoring:** Classifies each object into three behavioral states:
  * **Active (Green):** Normal movement.
  * **Stationary (Yellow):** No movement for > 150 frames (Resting).
  * **Dead (Red):** Potential mortality detected (> 450 frames).
* **State Recovery Logic:** Minimizes False Positives by restoring status to 'Active' immediately upon detected movement.
* **Data Logging:** Automatically exports frame-by-frame status data to `chicken_status_log.csv`.

[Image of a state machine diagram showing transitions between active, resting, and mortality states for animal behavior analysis]

## 🔬 Methodology
The system calculates the **Euclidean Distance ($d$)** between centroids in consecutive frames.

$$d = \sqrt{(x_{t+1}-x_t)^2 + (y_{t+1}-y_t)^2}$$

If the distance $d$ remains below the threshold $\epsilon$ for a specific number of frames, the system triggers a warning or mortality alert.

## ⚠️ Limitations & Future Work
현재 시스템의 한계를 인지하고 있으며, 다음과 같은 고도화 계획을 수립하였습니다:

1.  **False Positives:** 수면 중인 개체와 폐사체의 구분을 정교화하기 위해 **Optical Flow** 알고리즘 도입 예정 (미세 호흡 감지).
2.  **Occlusion:** 개체 간 겹침 현상 발생 시 ID Switching을 최소화하기 위한 Re-identification 로직 보완 필요.
3.  **Dynamic Thresholds:** 시간대별 활동 가중치를 적용한 가변 임계값 시스템 구축 예정.

[Image of a data analysis dashboard for livestock monitoring showing activity levels and mortality alerts]

## 📂 Repository Structure
* `src/`: Core source codes (`tracker.py`, `movement_analysis.py`).
* `data/`: (Private) Directory for video and model weights.
* `results/`: Sample logs and demonstration outputs.

## 🛠️ Installation
```bash
# Clone this repository
git clone [https://github.com/THENAEUN/Dead-Chicken-Detection.git](https://github.com/THENAEUN/Dead-Chicken-Detection.git)

# Install required libraries
pip install -r requirements.txt

