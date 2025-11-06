from itertools import combinations
from typing import Dict, Callable, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from artemis._utilities.domain import ProgressInfoLog

from artemis._utilities.ops import get_predict_function

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


# Ray task 함수들
if RAY_AVAILABLE:
    @ray.remote
    def compute_pd_single_group(X_ref, model_ref, predict_function, feature_group, pd_single_info, X_len, batchsize):
        """특성 그룹의 PD 계산 (Ray task)"""
        # Ray가 자동으로 ObjectRef를 역참조하므로 ray.get() 불필요
        X = X_ref
        model = model_ref
        
        range_dict = {}
        current_len = 0
        X_full = pd.DataFrame()
        results = {}
        
        for feature, f_values in feature_group:
            for value in f_values:
                change_dict = {feature: value}
                X_changed = X.copy().assign(**change_dict)
                range_dict[(feature, value)] = (current_len, current_len + X_len)
                current_len += X_len
                X_full = pd.concat((X_full, X_changed))
                
                if current_len >= batchsize:
                    y = predict_function(model, X_full)
                    for var_name, var_val in range_dict.keys():
                        start, end = range_dict[(var_name, var_val)]
                        results[(var_name, var_val)] = np.mean(y[start:end])
                    current_len = 0
                    range_dict = {}
                    X_full = pd.DataFrame()
        
        # 남은 배치 처리
        if current_len > 0:
            y = predict_function(model, X_full)
            for var_name, var_val in range_dict.keys():
                start, end = range_dict[(var_name, var_val)]
                results[(var_name, var_val)] = np.mean(y[start:end])
        
        return results

    @ray.remote
    def compute_pd_pairs_group(X_ref, model_ref, predict_function, pair_group, pd_pairs_info, X_len, batchsize, all_combinations):
        """특성 쌍 그룹의 PD 계산 (Ray task)"""
        # Ray가 자동으로 ObjectRef를 역참조하므로 ray.get() 불필요
        X = X_ref
        model = model_ref
        
        range_dict = {}
        current_len = 0
        X_full = pd.DataFrame()
        results = {}
        
        for feature1, feature2, f1_values, f2_values, pd_values in pair_group:
            if all_combinations:
                feature_values = [(f1, f2) for f1 in f1_values for f2 in f2_values]
            else:
                feature_values = list(zip(X[feature1].values, X[feature2].values))
            
            for value1, value2 in feature_values:
                f1_ind = get_index(f1_values, value1)
                f2_ind = get_index(f2_values, value2)
                
                if np.isnan(pd_values[f1_ind, f2_ind]):
                    change_dict = {feature1: value1, feature2: value2}
                    X_changed = X.copy().assign(**change_dict)
                    range_dict[(feature1, feature2, value1, value2)] = (current_len, current_len + X_len)
                    current_len += X_len
                    X_full = pd.concat((X_full, X_changed))
                    
                    if current_len >= batchsize:
                        y = predict_function(model, X_full)
                        for var_name1, var_name2, var_val1, var_val2 in range_dict.keys():
                            start, end = range_dict[(var_name1, var_name2, var_val1, var_val2)]
                            results[(var_name1, var_name2, var_val1, var_val2)] = np.mean(y[start:end])
                        current_len = 0
                        range_dict = {}
                        X_full = pd.DataFrame()
        
        # 남은 배치 처리
        if current_len > 0:
            y = predict_function(model, X_full)
            for var_name1, var_name2, var_val1, var_val2 in range_dict.keys():
                start, end = range_dict[(var_name1, var_name2, var_val1, var_val2)]
                results[(var_name1, var_name2, var_val1, var_val2)] = np.mean(y[start:end])
        
        return results

    @ray.remote
    def compute_pd_minus_single_group(X_ref, model_ref, predict_function, feature_group, X_len, batchsize):
        """특성 그룹의 PD-minus-single 계산 (Ray task)"""
        # Ray가 자동으로 ObjectRef를 역참조하므로 ray.get() 불필요
        X = X_ref
        model = model_ref
        
        range_dict = {}
        current_len = 0
        X_full = pd.DataFrame()
        results = {}
        
        for feature in feature_group:
            for i, row in X.copy().reset_index(drop=True).iterrows():
                change_dict = {other_feature: row[other_feature] for other_feature in X.columns if other_feature != feature}
                X_changed = X.copy().assign(**change_dict)
                range_dict[(feature, i)] = (current_len, current_len + X_len)
                current_len += X_len
                X_full = pd.concat((X_full, X_changed))
                
                if current_len >= batchsize:
                    y = predict_function(model, X_full)
                    for var_name, row_id in range_dict.keys():
                        start, end = range_dict[(var_name, row_id)]
                        results[(var_name, row_id)] = np.mean(y[start:end])
                    current_len = 0
                    range_dict = {}
                    X_full = pd.DataFrame()
        
        # 남은 배치 처리
        if current_len > 0:
            y = predict_function(model, X_full)
            for var_name, row_id in range_dict.keys():
                start, end = range_dict[(var_name, row_id)]
                results[(var_name, row_id)] = np.mean(y[start:end])
        
        return results


class PartialDependenceCalculator: 
    def __init__(self, model, X: pd.DataFrame, predict_function: Optional[Callable] = None, batchsize: int = 2000, n_jobs: int = 1, auto_shutdown: bool = False):
        self.model = model
        self.predict_function = get_predict_function(model, predict_function)
        self.X = X
        self.X_len = len(self.X)
        self.batchsize = batchsize
        self.n_jobs = n_jobs
        self.auto_shutdown = auto_shutdown
        
        # Ray 설정
        self.use_ray = n_jobs > 1 and RAY_AVAILABLE
        self._ray_initialized_by_me = False
        if self.use_ray:
            if not ray.is_initialized():
                ray.init(num_cpus=n_jobs, ignore_reinit_error=True)
                self._ray_initialized_by_me = True
            self.model_ref = ray.put(model)
            self.X_ref = ray.put(X)
        else:
            self.model_ref = None
            self.X_ref = None
        self.pd_single = {col: 
                            {   
                                "f_values": np.sort(np.unique(self.X[col].values)), 
                                "pd_values": np.full(len(np.unique(self.X[col].values)), np.nan) 
                            }
                            for col in self.X.columns}
        self.pd_pairs = {(col1, col2): 
                            {   
                                "f1_values": np.sort(np.unique(self.X[col1].values)), 
                                "f2_values": np.sort(np.unique(self.X[col2].values)),
                                "pd_values": np.full((len(np.unique(self.X[col1].values)), len(np.unique(self.X[col2].values))), np.nan)
                            }
                            for col1, col2 in combinations(self.X.columns, 2)}
        self.pd_minus_single = {col:
                            {   
                                "row_ids": np.arange(self.X_len),
                                "pd_values": np.full(self.X_len, np.nan) 
                            }
                            for col in self.X.columns}

    def get_pd_single(self, feature: str, feature_values: Optional[List[Any]] = None) -> np.ndarray:
        if feature_values is None:
            return self.pd_single[feature]["pd_values"]
        selected_values = np.zeros(len(feature_values))
        for i, feature_value in enumerate(feature_values):
            f_index = get_index(self.pd_single[feature]["f_values"], feature_value)
            selected_values[i] = self.pd_single[feature]["pd_values"][f_index]
        return selected_values

    def get_pd_pairs(self, feature1: str, feature2: str, feature_values: Optional[List[Tuple[Any, Any]]] = None) -> np.ndarray:
        pair_key = self._get_pair_key((feature1, feature2))
        all_matrix = self.pd_pairs[pair_key]["pd_values"]
        if feature_values is None:
            return all_matrix
        if pair_key != (feature1, feature2):
            feature_values = reorder_pair_values(feature_values)
        selected_values = np.zeros(len(feature_values))
        for i, pair in enumerate(feature_values):
            f1_index = get_index(self.pd_pairs[pair_key]["f1_values"], pair[0])
            f2_index = get_index(self.pd_pairs[pair_key]["f2_values"], pair[1])
            selected_values[i] = all_matrix[f1_index, f2_index]
        return selected_values
    
    def get_pd_minus_single(self, feature: str) -> np.ndarray:
        return self.pd_minus_single[feature]["pd_values"]

    def calculate_pd_single(self, features: Optional[List[str]] = None, show_progress: bool = False, desc: str = ProgressInfoLog.CALC_VAR_IMP):
        if features is None:
            features = self.X.columns
        
        if self.use_ray:
            self._calculate_pd_single_ray(features, show_progress, desc)
        else:
            self._calculate_pd_single_sequential(features, show_progress, desc)
    
    def _calculate_pd_single_sequential(self, features, show_progress: bool, desc: str):
        """순차 처리 버전 (원래 로직)"""
        range_dict = {}
        current_len = 0
        X_full = pd.DataFrame()
        for feature in tqdm(features, desc=desc, disable=not show_progress):
            if np.isnan(self.pd_single[feature]["pd_values"]).any(): 
                for value in self.pd_single[feature]["f_values"]:
                    change_dict = {feature: value}
                    X_changed = self.X.copy().assign(**change_dict)
                    range_dict[(feature, value)] = (current_len, current_len+self.X_len)
                    current_len += self.X_len
                    X_full = pd.concat((X_full, X_changed))
                if current_len > self.batchsize:
                    self.fill_pd_single(range_dict, X_full)
                    current_len = 0
                    range_dict = {}
                    X_full = pd.DataFrame()
        if current_len > 0:
            self.fill_pd_single(range_dict, X_full)
    
    def _calculate_pd_single_ray(self, features, show_progress: bool, desc: str):
        """Ray 병렬 처리 버전"""
        # 계산이 필요한 특성 필터링
        features_to_compute = [f for f in features if np.isnan(self.pd_single[f]["pd_values"]).any()]
        
        if not features_to_compute:
            return
        
        # 특성을 그룹으로 분할
        feature_groups = []
        group_size = max(1, len(features_to_compute) // self.n_jobs)
        for i in range(0, len(features_to_compute), group_size):
            group = [(f, self.pd_single[f]["f_values"]) for f in features_to_compute[i:i+group_size]]
            feature_groups.append(group)
        
        # Ray task 실행
        futures = [
            compute_pd_single_group.remote(
                self.X_ref, self.model_ref, self.predict_function,
                group, None, self.X_len, self.batchsize
            )
            for group in feature_groups
        ]
        
        # 진행 상황 표시와 함께 결과 수집
        completed = []
        if show_progress:
            with tqdm(total=len(futures), desc=desc) as pbar:
                while futures:
                    ready, futures = ray.wait(futures, num_returns=1)
                    completed.extend(ready)
                    pbar.update(1)
        else:
            completed = futures
        
        all_results = ray.get(completed)
        
        # 결과 병합
        for results in all_results:
            for (var_name, var_val), pd_value in results.items():
                value_index = get_index(self.pd_single[var_name]["f_values"], var_val)
                self.pd_single[var_name]["pd_values"][value_index] = pd_value
    
    def calculate_pd_pairs(self, feature_pairs = None, all_combinations=True, show_progress: bool = False, desc: str = ProgressInfoLog.CALC_OVO):
        if feature_pairs is None:
            feature_pairs = self.pd_pairs.keys()
        
        if self.use_ray:
            self._calculate_pd_pairs_ray(feature_pairs, all_combinations, show_progress, desc)
        else:
            self._calculate_pd_pairs_sequential(feature_pairs, all_combinations, show_progress, desc)
    
    def _calculate_pd_pairs_sequential(self, feature_pairs, all_combinations: bool, show_progress: bool, desc: str):
        """순차 처리 버전 (원래 로직)"""
        range_dict = {}
        current_len = 0
        X_full = pd.DataFrame()
        for feature1, feature2 in tqdm(feature_pairs, desc=desc, disable=not show_progress):
            feature1, feature2 = self._get_pair_key((feature1, feature2))
            if all_combinations:
                feature_values = [(f1, f2) for f1 in self.pd_pairs[(feature1, feature2)]["f1_values"] for f2 in self.pd_pairs[(feature1, feature2)]["f2_values"]]
            else:
                feature_values = zip(self.X[feature1].values, self.X[feature2].values)         
            for value1, value2 in feature_values:
                f1_ind = get_index(self.pd_pairs[(feature1, feature2)]["f1_values"], value1)
                f2_ind = get_index(self.pd_pairs[(feature1, feature2)]["f2_values"], value2)
                if np.isnan(self.pd_pairs[(feature1, feature2)]["pd_values"][f1_ind, f2_ind]): 
                    change_dict = {feature1: value1, feature2: value2}
                    X_changed = self.X.copy().assign(**change_dict)
                    range_dict[(feature1, feature2, value1, value2)] = (current_len, current_len+self.X_len)
                    current_len += self.X_len
                    X_full = pd.concat((X_full, X_changed))
                    if current_len > self.batchsize:
                        self.fill_pd_pairs(range_dict, X_full)
                        current_len = 0
                        range_dict = {}
                        X_full = pd.DataFrame()
        if current_len > 0:
            self.fill_pd_pairs(range_dict, X_full)
    
    def _calculate_pd_pairs_ray(self, feature_pairs, all_combinations: bool, show_progress: bool, desc: str):
        """Ray 병렬 처리 버전"""
        # 특성 쌍을 정규화하고 그룹으로 분할
        pairs_to_compute = []
        for feature1, feature2 in feature_pairs:
            feature1, feature2 = self._get_pair_key((feature1, feature2))
            pairs_to_compute.append((
                feature1, feature2,
                self.pd_pairs[(feature1, feature2)]["f1_values"],
                self.pd_pairs[(feature1, feature2)]["f2_values"],
                self.pd_pairs[(feature1, feature2)]["pd_values"]
            ))
        
        if not pairs_to_compute:
            return
        
        # 특성 쌍을 그룹으로 분할
        pair_groups = []
        group_size = max(1, len(pairs_to_compute) // self.n_jobs)
        for i in range(0, len(pairs_to_compute), group_size):
            pair_groups.append(pairs_to_compute[i:i+group_size])
        
        # Ray task 실행
        futures = [
            compute_pd_pairs_group.remote(
                self.X_ref, self.model_ref, self.predict_function,
                group, None, self.X_len, self.batchsize, all_combinations
            )
            for group in pair_groups
        ]
        
        # 진행 상황 표시와 함께 결과 수집
        completed = []
        if show_progress:
            with tqdm(total=len(futures), desc=desc) as pbar:
                while futures:
                    ready, futures = ray.wait(futures, num_returns=1)
                    completed.extend(ready)
                    pbar.update(1)
        else:
            completed = futures
        
        all_results = ray.get(completed)
        
        # 결과 병합
        for results in all_results:
            for (var_name1, var_name2, var_val1, var_val2), pd_value in results.items():
                value_index1 = get_index(self.pd_pairs[(var_name1, var_name2)]["f1_values"], var_val1)
                value_index2 = get_index(self.pd_pairs[(var_name1, var_name2)]["f2_values"], var_val2)
                self.pd_pairs[(var_name1, var_name2)]["pd_values"][value_index1, value_index2] = pd_value
    
    def calculate_pd_minus_single(self, features: Optional[List[str]] = None, show_progress: bool = False, desc: str = ProgressInfoLog.CALC_OVA):
        if features is None:
            features = self.X.columns
        
        if self.use_ray:
            self._calculate_pd_minus_single_ray(features, show_progress, desc)
        else:
            self._calculate_pd_minus_single_sequential(features, show_progress, desc)
    
    def _calculate_pd_minus_single_sequential(self, features, show_progress: bool, desc: str):
        """순차 처리 버전 (원래 로직)"""
        range_dict = {}
        current_len = 0
        X_full = pd.DataFrame()
        for feature in tqdm(features, desc=desc, disable=not show_progress):
            if np.isnan(self.pd_minus_single[feature]["pd_values"]).any(): 
                for i, row in self.X.copy().reset_index(drop=True).iterrows():
                    change_dict = {other_feature: row[other_feature] for other_feature in self.X.columns if other_feature != feature}
                    X_changed = self.X.copy().assign(**change_dict)
                    range_dict[(feature, i)] = (current_len, current_len+self.X_len)
                    current_len += self.X_len
                    X_full = pd.concat((X_full, X_changed))
                if current_len > self.batchsize:
                    self.fill_pd_minus_single(range_dict, X_full)
                    current_len = 0
                    range_dict = {}
                    X_full = pd.DataFrame()
        if current_len > 0:
            self.fill_pd_minus_single(range_dict, X_full)
    
    def _calculate_pd_minus_single_ray(self, features, show_progress: bool, desc: str):
        """Ray 병렬 처리 버전"""
        # 계산이 필요한 특성 필터링
        features_to_compute = [f for f in features if np.isnan(self.pd_minus_single[f]["pd_values"]).any()]
        
        if not features_to_compute:
            return
        
        # 특성을 그룹으로 분할
        feature_groups = []
        group_size = max(1, len(features_to_compute) // self.n_jobs)
        for i in range(0, len(features_to_compute), group_size):
            feature_groups.append(features_to_compute[i:i+group_size])
        
        # Ray task 실행
        futures = [
            compute_pd_minus_single_group.remote(
                self.X_ref, self.model_ref, self.predict_function,
                group, self.X_len, self.batchsize
            )
            for group in feature_groups
        ]
        
        # 진행 상황 표시와 함께 결과 수집
        completed = []
        if show_progress:
            with tqdm(total=len(futures), desc=desc) as pbar:
                while futures:
                    ready, futures = ray.wait(futures, num_returns=1)
                    completed.extend(ready)
                    pbar.update(1)
        else:
            completed = futures
        
        all_results = ray.get(completed)
        
        # 결과 병합
        for results in all_results:
            for (var_name, row_id), pd_value in results.items():
                self.pd_minus_single[var_name]["pd_values"][row_id] = pd_value

    def fill_pd_single(self, range_dict, X_full): 
        y = self.predict_function(self.model, X_full)
        for var_name, var_val in range_dict.keys():
            start, end = range_dict[(var_name, var_val)]
            value_index = get_index(self.pd_single[var_name]["f_values"], var_val)
            self.pd_single[var_name]["pd_values"][value_index] = np.mean(y[start:end])

    def fill_pd_pairs(self, range_dict, X_full):
        y = self.predict_function(self.model, X_full)
        for var_name1, var_name2, var_val1, var_val2 in range_dict.keys():
            start, end = range_dict[(var_name1, var_name2, var_val1, var_val2)]
            value_index1 = get_index(self.pd_pairs[(var_name1, var_name2)]["f1_values"], var_val1)
            value_index2 = get_index(self.pd_pairs[(var_name1, var_name2)]["f2_values"], var_val2)
            self.pd_pairs[(var_name1, var_name2)]["pd_values"][value_index1, value_index2] = np.mean(y[start:end])

    def fill_pd_minus_single(self, range_dict, X_full): 
        y = self.predict_function(self.model, X_full)
        for var_name, row_id in range_dict.keys():
            start, end = range_dict[(var_name, row_id)]
            self.pd_minus_single[var_name]["pd_values"][row_id] = np.mean(y[start:end])

    def _get_pair_key(self, pair: Tuple[str, str]) -> Tuple[str, str]:
        if pair in self.pd_pairs.keys():
            return pair
        else:
            return (pair[1], pair[0])
    
    def cleanup(self):
        """Ray 리소스 정리
        
        Ray를 사용한 경우 리소스를 정리합니다.
        auto_shutdown=True인 경우 자동으로 호출되지만, 
        명시적으로 호출할 수도 있습니다.
        """
        if self.use_ray and self._ray_initialized_by_me and RAY_AVAILABLE:
            if ray.is_initialized():
                ray.shutdown()
                self._ray_initialized_by_me = False
    
    def __enter__(self):
        """Context manager 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료 시 자동 cleanup"""
        if self.auto_shutdown or self._ray_initialized_by_me:
            self.cleanup()
        return False
    
    def __del__(self):
        """소멸자: 객체가 삭제될 때 호출"""
        if self.auto_shutdown:
            self.cleanup()


def get_index(array, value) -> int:
    return np.where(array == value)[0][0]

def reorder_pair_values(pair_values: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
    return [(pair[1], pair[0]) for pair in pair_values]


