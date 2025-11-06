#!/usr/bin/env python
"""
포크한 저장소에서 설치한 artemis 패키지 테스트
"""

def test_fork_installation():
    """포크한 저장소에서 설치한 패키지가 정상 작동하는지 테스트"""
    print("=" * 60)
    print("포크한 저장소에서 설치한 artemis 패키지 테스트")
    print("=" * 60)
    
    # Import 테스트
    try:
        import artemis
        print(f'✓ artemis 패키지 import 성공')
        print(f'  위치: {artemis.__file__}')
    except Exception as e:
        print(f'✗ artemis 패키지 import 실패: {e}')
        return False
    
    # 주요 클래스들 import 테스트
    try:
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
        print('✓ 모든 모듈 import 성공')
    except Exception as e:
        print(f'✗ 모듈 import 실패: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    # 버전 확인
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        print(f'\n✓ 의존성 패키지 버전 확인:')
        print(f'  pandas: {pd.__version__}')
        print(f'  numpy: {np.__version__}')
        print(f'  scikit-learn: {sklearn.__version__}')
    except Exception as e:
        print(f'⚠ 의존성 버전 확인 실패: {e}')
    
    print("\n" + "=" * 60)
    print("모든 테스트가 성공했습니다! ✓")
    print("=" * 60)
    print("\n📦 설치 정보:")
    print("  저장소: https://github.com/mediummandoo/artemis")
    print("  브랜치: fix-dependencies2511")
    print("  설치 명령어: pip install git+https://github.com/mediummandoo/artemis.git@fix-dependencies2511")
    return True

if __name__ == "__main__":
    success = test_fork_installation()
    exit(0 if success else 1)

