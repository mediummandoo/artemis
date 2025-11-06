# Installation Guide / 설치 가이드

## English

### Installing from GitHub Fork

#### Step 1: Fork the Repository

1. Visit the original repository: https://github.com/pyartemis/artemis
2. Click the "Fork" button in the top right corner
3. The repository will be copied to your GitHub account

#### Step 2: Install from Your Fork

##### Google Colab

```python
# Install from your fork
!pip install git+https://github.com/mediummandoo/artemis.git

# Install specific commit (recommended for stability)
!pip install git+https://github.com/mediummandoo/artemis.git@COMMIT-HASH

# Install specific branch
!pip install git+https://github.com/mediummandoo/artemis.git@BRANCH-NAME
```

##### Local Installation

```bash
# Install from GitHub fork
pip install git+https://github.com/mediummandoo/artemis.git

# Install specific commit (recommended)
pip install git+https://github.com/mediummandoo/artemis.git@COMMIT-HASH

# Install in editable mode (for development)
git clone https://github.com/mediummandoo/artemis.git
cd artemis
pip install -e .
```

##### Using requirements.txt

```txt
# requirements.txt
git+https://github.com/mediummandoo/artemis.git@COMMIT-HASH
```

Then install:
```bash
pip install -r requirements.txt
```

#### Step 3: Verify Installation

```python
import artemis
from artemis.additivity import AdditivityMeter
print("✓ Installation successful!")
```

---

## 한국어

### GitHub 포크에서 설치하기

#### 1단계: 저장소 포크하기

1. 원본 저장소 방문: https://github.com/pyartemis/artemis
2. 우측 상단의 "Fork" 버튼 클릭
3. 저장소가 본인의 GitHub 계정으로 복사됩니다

#### 2단계: 포크한 저장소에서 설치하기

##### Google Colab

```python
# 포크한 저장소에서 설치
!pip install git+https://github.com/mediummandoo/artemis.git

# 특정 커밋 설치 (안정성을 위해 권장)
!pip install git+https://github.com/mediummandoo/artemis.git@커밋해시

# 특정 브랜치 설치
!pip install git+https://github.com/mediummandoo/artemis.git@브랜치명
```

##### 로컬 설치

```bash
# GitHub 포크에서 설치
pip install git+https://github.com/mediummandoo/artemis.git

# 특정 커밋 설치 (권장)
pip install git+https://github.com/mediummandoo/artemis.git@커밋해시

# 개발 모드로 설치 (코드 수정 가능)
git clone https://github.com/mediummandoo/artemis.git
cd artemis
pip install -e .
```

##### requirements.txt 사용

```txt
# requirements.txt
git+https://github.com/mediummandoo/artemis.git@커밋해시
```

설치:
```bash
pip install -r requirements.txt
```

#### 3단계: 설치 확인

```python
import artemis
from artemis.additivity import AdditivityMeter
print("✓ 설치 성공!")
```
