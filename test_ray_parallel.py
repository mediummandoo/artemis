"""
Ray 병렬화 테스트 스크립트

순차 처리와 병렬 처리의 성능을 비교합니다.
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Artemis import
from artemis.interactions_methods.model_agnostic import (
    FriedmanHStatisticMethod,
    GreenwellMethod,
    SejongOhMethod
    # SejongOhMethod may be part of performance_based methods; import with fallback
)
from artemis.interactions_methods.model_specific import (
    ConditionalMinimalDepthMethod,
    SplitScoreMethod,
)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def create_test_data():
    """테스트 데이터 생성"""
    from sklearn.datasets import fetch_california_housing
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    return X, y


def check_reproducibility(method_seq, method_par, method_name):
    """재현성 체크: 순차 vs 병렬 결과 비교"""
    
    # 1. OVO 결과 비교
    print(f"1️⃣  {method_name} OVO 결과 일치 여부:")
    ovo_seq = method_seq.ovo.sort_values(by=["Feature 1", "Feature 2"]).reset_index(drop=True)
    ovo_par = method_par.ovo.sort_values(by=["Feature 1", "Feature 2"]).reset_index(drop=True)
    
    # 값 비교 (상호작용 강도)
    values_match = np.allclose(
        ovo_seq[method_seq.method].values, 
        ovo_par[method_par.method].values, 
        rtol=1e-10, atol=1e-10
    )
    
    if values_match:
        print(f"   ✅ OVO 값이 정확히 일치합니다!")
        max_diff = np.max(np.abs(ovo_seq[method_seq.method].values - ovo_par[method_par.method].values))
        print(f"   최대 차이: {max_diff:.2e}")
    else:
        print(f"   ⚠️  OVO 값에 차이가 있습니다")
        diff = np.abs(ovo_seq[method_seq.method].values - ovo_par[method_par.method].values)
        print(f"   최대 차이: {np.max(diff):.6f}")
        print(f"   평균 차이: {np.mean(diff):.6f}")
    
    # 2. Feature Importance 비교
    print(f"\n2️⃣  {method_name} Feature Importance 일치 여부:")
    imp_seq = method_seq.feature_importance.sort_values(by="Feature").reset_index(drop=True)
    imp_par = method_par.feature_importance.sort_values(by="Feature").reset_index(drop=True)
    
    imp_match = np.allclose(
        imp_seq["Importance"].values,
        imp_par["Importance"].values,
        rtol=1e-10, atol=1e-10
    )
    
    if imp_match:
        print(f"   ✅ Feature Importance가 정확히 일치합니다!")
        max_diff = np.max(np.abs(imp_seq["Importance"].values - imp_par["Importance"].values))
        print(f"   최대 차이: {max_diff:.2e}")
    else:
        print(f"   ⚠️  Feature Importance에 차이가 있습니다")
        diff = np.abs(imp_seq["Importance"].values - imp_par["Importance"].values)
        print(f"   최대 차이: {np.max(diff):.6f}")
        print(f"   평균 차이: {np.mean(diff):.6f}")
    
    # 3. 전체 결론
    if values_match and imp_match:
        print(f"\n✅ {method_name}: 순차와 병렬 처리 결과가 완벽히 일치합니다!")
        return True
    else:
        print(f"\n⚠️  {method_name}: 순차와 병렬 처리에서 미세한 차이가 발견되었습니다")
        return False


def test_multiple_runs_reproducibility():
    """동일 설정으로 여러 번 실행 시 재현성 테스트"""
    print("\n" + "=" * 60)
    print("재현성 테스트: 동일 설정으로 반복 실행")
    print("=" * 60)
    
    # 데이터 및 모델 준비
    X, y = create_test_data()
    model = RandomForestRegressor(n_estimators=80, max_depth=4, max_features="sqrt", random_state=42)
    model.fit(X, y)
    
    print("\n[병렬 처리를 3회 반복 실행]")
    
    results = []
    for i in range(3):
        print(f"\n실행 {i+1}/3...")
        method = GreenwellMethod(random_state=42)
        method.fit(model, X, n=200, show_progress=False, n_jobs=4)
        results.append({
            'ovo': method.ovo.copy(),
            'importance': method.feature_importance.copy()
        })
    
    # 결과 비교
    print("\n[재현성 분석]")
    
    # Run 1 vs Run 2
    ovo_1 = results[0]['ovo'].sort_values(by=["Feature 1", "Feature 2"]).reset_index(drop=True)
    ovo_2 = results[1]['ovo'].sort_values(by=["Feature 1", "Feature 2"]).reset_index(drop=True)
    ovo_3 = results[2]['ovo'].sort_values(by=["Feature 1", "Feature 2"]).reset_index(drop=True)
    
    match_12 = np.allclose(
        ovo_1[results[0]['ovo'].columns[-1]].values,
        ovo_2[results[1]['ovo'].columns[-1]].values,
        rtol=1e-10, atol=1e-10
    )
    
    match_23 = np.allclose(
        ovo_2[results[1]['ovo'].columns[-1]].values,
        ovo_3[results[2]['ovo'].columns[-1]].values,
        rtol=1e-10, atol=1e-10
    )
    
    if match_12 and match_23:
        print("✅ 3회 실행 결과가 모두 동일합니다!")
        print("   병렬 처리에서도 완벽한 재현성이 보장됩니다.")
    else:
        print("⚠️  실행 결과에 차이가 있습니다")
        if not match_12:
            diff = np.abs(ovo_1[results[0]['ovo'].columns[-1]].values - 
                         ovo_2[results[1]['ovo'].columns[-1]].values)
            print(f"   Run1 vs Run2 최대 차이: {np.max(diff):.6f}")
        if not match_23:
            diff = np.abs(ovo_2[results[1]['ovo'].columns[-1]].values - 
                         ovo_3[results[2]['ovo'].columns[-1]].values)
            print(f"   Run2 vs Run3 최대 차이: {np.max(diff):.6f}")
    
    return match_12 and match_23


def test_friedman_sequential_vs_parallel():
    """Friedman H-Statistic 순차 vs 병렬 비교"""
    print("=" * 60)
    print("Friedman H-Statistic: 순차 vs 병렬 처리 비교")
    print("=" * 60)
    
    # 데이터 및 모델 준비
    X, y = create_test_data()
    model = RandomForestRegressor(n_estimators=80, max_depth=4, max_features="sqrt", random_state=42)
    model.fit(X, y)
    
    # 순차 처리 (n_jobs=1)
    print("\n[순차 처리] n_jobs=1")
    method_seq = FriedmanHStatisticMethod(random_state=42)
    start_time = time.time()
    method_seq.fit(model, X, n=200, show_progress=True, n_jobs=1)
    seq_time = time.time() - start_time
    print(f"실행 시간: {seq_time:.2f}초")
    
    # 병렬 처리 (n_jobs=4)
    print("\n[병렬 처리] n_jobs=4")
    method_par = FriedmanHStatisticMethod(random_state=42)
    start_time = time.time()
    method_par.fit(model, X, n=200, show_progress=True, n_jobs=4)
    par_time = time.time() - start_time
    print(f"실행 시간: {par_time:.2f}초")
    
    # 성능 비교
    speedup = seq_time / par_time
    print(f"\n성능 향상: {speedup:.2f}x")
    
    # 결과 비교
    print("\n[결과 비교]")
    print("OVO 상호작용 (상위 5개):")
    print(method_par.ovo.head())
    print("\n특성 중요도:")
    print(method_par.feature_importance)
    
    # 재현성 체크
    print("\n[재현성 체크]")
    check_reproducibility(method_seq, method_par, "Friedman H-Statistic")


def test_greenwell_parallel():
    """Greenwell Method 병렬 처리 테스트"""
    print("\n" + "=" * 60)
    print("Greenwell Method: 순차 vs 병렬 처리 비교")
    print("=" * 60)
    
    # 데이터 및 모델 준비
    X, y = create_test_data()
    model = RandomForestRegressor(n_estimators=80, max_depth=4, max_features="sqrt", random_state=42)
    model.fit(X, y)
    
    # 순차 처리
    print("\n[순차 처리] n_jobs=1")
    method_seq = GreenwellMethod(random_state=42)
    start_time = time.time()
    method_seq.fit(model, X, n=200, show_progress=True, n_jobs=1)
    seq_time = time.time() - start_time
    print(f"실행 시간: {seq_time:.2f}초")
    
    # 병렬 처리
    print("\n[병렬 처리] n_jobs=4")
    method_par = GreenwellMethod(random_state=42)
    start_time = time.time()
    method_par.fit(model, X, n=200, show_progress=True, n_jobs=4)
    par_time = time.time() - start_time
    print(f"실행 시간: {par_time:.2f}초")
    
    # 성능 비교
    speedup = seq_time / par_time
    print(f"\n성능 향상: {speedup:.2f}x")
    
    print("\n[결과]")
    print("OVO 상호작용 (상위 5개):")
    print(method_par.ovo.head())
    print("\n특성 중요도:")
    print(method_par.feature_importance)
    
    # 재현성 체크
    print("\n[재현성 체크]")
    check_reproducibility(method_seq, method_par, "Greenwell Method")


def test_importance_parallel():
    """Feature Importance 병렬 처리 테스트"""
    print("\n" + "=" * 60)
    print("Feature Importance: 병렬 처리 테스트")
    print("=" * 60)
    
    # 데이터 및 모델 준비
    X, y = create_test_data()
    model = RandomForestRegressor(n_estimators=80, max_depth=4, max_features="sqrt", random_state=42)
    model.fit(X, y)
    from artemis.importance_methods.model_agnostic import PartialDependenceBasedImportance
    # 병렬 처리
    print("\n[병렬 처리] n_jobs=4")
    method = PartialDependenceBasedImportance()
    start_time = time.time()
    importance = method.importance(model, X, n=200, show_progress=True, n_jobs=4)
    exec_time = time.time() - start_time
    print(f"실행 시간: {exec_time:.2f}초")
    
    print("\n[결과]")
    print(importance)


def test_pd_calculator_reuse():
    """PD Calculator 재사용 테스트"""
    print("\n" + "=" * 60)
    print("PD Calculator 재사용 테스트")
    print("=" * 60)
    
    from artemis._utilities.pd_calculator import PartialDependenceCalculator
    
    # 데이터 및 모델 준비
    X, y = create_test_data()
    model = RandomForestRegressor(n_estimators=80, max_depth=4, max_features="sqrt", random_state=42)
    model.fit(X, y)
    
    # PD Calculator 생성 (병렬)
    print("\n[PD Calculator 생성] n_jobs=4")
    start_time = time.time()
    pd_calc = PartialDependenceCalculator(
        model, X[:200], 
        batchsize=2000, 
        n_jobs=4
    )
    create_time = time.time() - start_time
    print(f"생성 시간: {create_time:.2f}초")
    
    # 여러 메소드에서 재사용
    print("\n[Friedman - PD Calculator 재사용]")
    method1 = FriedmanHStatisticMethod(random_state=42)
    start_time = time.time()
    method1.fit(model, X[:200], pd_calculator=pd_calc, show_progress=True)
    method1_time = time.time() - start_time
    print(f"실행 시간: {method1_time:.2f}초")
    
    print("\n[Greenwell - PD Calculator 재사용]")
    method2 = GreenwellMethod(random_state=42)
    start_time = time.time()
    method2.fit(model, X[:200], pd_calculator=pd_calc, show_progress=True)
    method2_time = time.time() - start_time
    print(f"실행 시간: {method2_time:.2f}초")
    
    print(f"\n총 시간: {create_time + method1_time + method2_time:.2f}초")


def main():
    """메인 테스트 실행"""
    print("\n🚀 Artemis Ray 병렬화 테스트 시작\n")
    
    try:
        import ray
        print(f"✅ Ray 버전: {ray.__version__}\n")
    except ImportError:
        print("❌ Ray가 설치되지 않았습니다. 'pip install ray'로 설치하세요.\n")
        return
    
    try:
        # 테스트 실행
        test_friedman_sequential_vs_parallel()
        test_greenwell_parallel()
        test_importance_parallel()
        test_pd_calculator_reuse()
        test_multiple_runs_reproducibility()
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ray 정리
        if ray.is_initialized():
            ray.shutdown()
            print("\n🧹 Ray 리소스 정리 완료")


if __name__ == "__main__":
    main()

