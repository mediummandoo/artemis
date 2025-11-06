# ARTEMIS 패키지 기능 목록

## 개요
ARTEMIS는 머신러닝 모델에서 특징(feature) 간 상호작용(interactions)을 추출하기 위한 설명 가능한 AI(XAI) 방법들을 제공하는 Python 패키지입니다.

## 주요 기능 카테고리

### 1. 가산성 측정 (Additivity)
- **AdditivityMeter**: 모델의 가산성 지수를 계산하여 모델이 얼마나 가산적인지 측정

### 2. 방법 비교 (Comparison)
- **FeatureInteractionMethodComparator**: 서로 다른 상호작용 방법들을 비교하는 기능

### 3. 변수 중요도 방법 (Importance Methods)

#### 모델 비의존적 (Model Agnostic)
- **PermutationImportance**: 순열 기반 변수 중요도
- **PartialDependenceBasedImportance**: 부분 의존성 기반 변수 중요도

#### 모델 특화 (Model Specific)
- **MinimalDepthImportance**: 최소 깊이 기반 변수 중요도 (트리 기반 모델용)
- **SplitScoreImportance**: 분할 점수 기반 변수 중요도 (트리 기반 모델용)

### 4. 상호작용 방법 (Interaction Methods)

#### 모델 비의존적 (Model Agnostic)

**부분 의존성 기반 (Partial Dependence Based)**
- **FriedmanHStatisticMethod**: Friedman H-통계량 상호작용 측정
- **GreenwellMethod**: Greenwell 변수 상호작용 측정

**성능 기반 (Performance Based)**
- **SejongOhMethod**: Sejong Oh 성능 기반 상호작용 측정

#### 모델 특화 (Model Specific)

**그래디언트 부스팅 트리 (GB Trees)**
- **SplitScoreMethod**: 분할 점수 기반 상호작용 측정

**랜덤 포레스트 (Random Forest)**
- **ConditionalMinimalDepthMethod**: 조건부 최소 깊이 기반 상호작용 측정

### 5. 시각화 (Visualization)
- **PartialDependenceVisualizer**: 부분 의존성 플롯(Partial Dependence Plot) 시각화

### 6. 유틸리티 (Utilities)

도메인 정의 클래스들:
- **InteractionMethod**: 상호작용 방법 타입 정의
- **ImportanceMethod**: 중요도 방법 타입 정의
- **InteractionCalculationStrategy**: 상호작용 계산 전략 (ONE_VS_ONE, ONE_VS_ALL)
- **VisualizationType**: 시각화 타입 정의 (summary, graph, bar_chart, heatmap, lollipop 등)
- **ProblemType**: 문제 타입 정의 (REGRESSION, CLASSIFICATION)
- **CorrelationMethod**: 상관관계 방법 정의 (PEARSON, KENDALL, SPEARMAN)

## 사용 예시

```python
# Import 예시
from artemis.additivity import AdditivityMeter
from artemis.comparison import FeatureInteractionMethodComparator
from artemis.importance_methods.model_agnostic import (
    PermutationImportance, 
    PartialDependenceBasedImportance
)
from artemis.importance_methods.model_specific import (
    MinimalDepthImportance, 
    SplitScoreImportance
)
from artemis.interactions_methods.model_agnostic import (
    FriedmanHStatisticMethod, 
    GreenwellMethod, 
    SejongOhMethod
)
from artemis.interactions_methods.model_specific import (
    SplitScoreMethod, 
    ConditionalMinimalDepthMethod
)
from artemis.visualizer import PartialDependenceVisualizer
```

## 지원 데이터 타입
- 테이블 형식 데이터 (Tabular Data)
- 분류 문제 (Classification)
- 회귀 문제 (Regression)

## 지원 모델 타입
- 범용 모델 (모델 비의존적 방법)
- 트리 기반 모델 (Random Forest, Gradient Boosting Trees 등)

