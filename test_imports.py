#!/usr/bin/env python
"""
artemis 패키지의 모든 기능이 정상적으로 import되는지 확인하는 스크립트
"""

def test_all_imports():
    """모든 artemis 기능을 import하고 테스트합니다."""
    print("=" * 60)
    print("artemis 패키지 Import 테스트")
    print("=" * 60)
    
    try:
        import artemis
        print('✓ artemis 패키지 import 성공')
    except Exception as e:
        print(f'✗ artemis 패키지 import 실패: {e}')
        return False
    
    # Additivity
    try:
        from artemis.additivity import AdditivityMeter
        print('✓ AdditivityMeter import 성공')
    except Exception as e:
        print(f'✗ AdditivityMeter import 실패: {e}')
        return False
    
    # Comparison
    try:
        from artemis.comparison import FeatureInteractionMethodComparator
        print('✓ FeatureInteractionMethodComparator import 성공')
    except Exception as e:
        print(f'✗ FeatureInteractionMethodComparator import 실패: {e}')
        return False
    
    # Importance Methods - Model Agnostic
    try:
        from artemis.importance_methods.model_agnostic import (
            PermutationImportance, 
            PartialDependenceBasedImportance
        )
        print('✓ PermutationImportance, PartialDependenceBasedImportance import 성공')
    except Exception as e:
        print(f'✗ Importance Methods (Model Agnostic) import 실패: {e}')
        return False
    
    # Importance Methods - Model Specific
    try:
        from artemis.importance_methods.model_specific import (
            MinimalDepthImportance, 
            SplitScoreImportance
        )
        print('✓ MinimalDepthImportance, SplitScoreImportance import 성공')
    except Exception as e:
        print(f'✗ Importance Methods (Model Specific) import 실패: {e}')
        return False
    
    # Interaction Methods - Model Agnostic
    try:
        from artemis.interactions_methods.model_agnostic import (
            FriedmanHStatisticMethod, 
            GreenwellMethod, 
            SejongOhMethod
        )
        print('✓ FriedmanHStatisticMethod, GreenwellMethod, SejongOhMethod import 성공')
    except Exception as e:
        print(f'✗ Interaction Methods (Model Agnostic) import 실패: {e}')
        return False
    
    # Interaction Methods - Model Specific
    try:
        from artemis.interactions_methods.model_specific import (
            SplitScoreMethod, 
            ConditionalMinimalDepthMethod
        )
        print('✓ SplitScoreMethod, ConditionalMinimalDepthMethod import 성공')
    except Exception as e:
        print(f'✗ Interaction Methods (Model Specific) import 실패: {e}')
        return False
    
    # Visualizer
    try:
        from artemis.visualizer import PartialDependenceVisualizer
        print('✓ PartialDependenceVisualizer import 성공')
    except Exception as e:
        print(f'✗ PartialDependenceVisualizer import 실패: {e}')
        return False
    
    # Utilities
    try:
        from artemis._utilities.domain import (
            InteractionMethod, 
            ImportanceMethod, 
            InteractionCalculationStrategy, 
            VisualizationType, 
            ProblemType, 
            CorrelationMethod
        )
        print('✓ Utilities (domain) import 성공')
    except Exception as e:
        print(f'✗ Utilities import 실패: {e}')
        return False
    
    print("\n" + "=" * 60)
    print("모든 기능이 정상적으로 import되었습니다! ✓")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_all_imports()
    exit(0 if success else 1)

