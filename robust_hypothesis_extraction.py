# robust_hypothesis_extraction.py
import json
import os
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import heapq
import threading
import pickle
import numpy as np # Added for np.random in example

# Define a simple structure for HypothesisData if not already defined elsewhere
# from your_project_path.hypothesis import HypothesisData # Example
# For now, using a Dict as a placeholder for hypothesis_data
HypothesisData = Dict[str, Any]


class HypothesisTracker:
    def __init__(self,
                 max_hypotheses: int = 100000,
                 autosave_interval: int = 100, # Number of new hypotheses before auto-saving
                 save_directory: str = "hypothesis_tracking_data",
                 top_n_to_keep: int = 100): # Keep details for top N for quick access

        self.max_hypotheses = max_hypotheses
        self.autosave_interval = autosave_interval
        self.save_directory = save_directory
        self.top_n_to_keep = top_n_to_keep # For specific sorted lists

        os.makedirs(self.save_directory, exist_ok=True)

        self.hypotheses: Dict[str, Dict[str, Any]] = {}

        self.metadata: Dict[str, Any] = {
            'total_evaluated': 0,
            'last_saved_count': 0,
            'start_time': time.time(),
        }

        self.best_overall_heap: List[Tuple[float, str]] = []
        self.best_conservation_heap: List[Tuple[float, str]] = []
        self.best_symmetry_heap: List[Tuple[float, str]] = []
        self.best_trajectory_fit_heap: List[Tuple[float, str]] = []

        self._lock = threading.Lock()

        self.load_state()

    def _generate_id(self, hypothesis_data: HypothesisData) -> str:
        return str(uuid.uuid4())

    def add_hypothesis(self,
                       hypothesis_id: str,
                       hypothesis_data: HypothesisData,
                       evaluation_results: Dict[str, Any],
                       step: Optional[int] = None,
                       episode: Optional[int] = None,
                       training_context: Optional[Dict[str, Any]] = None):
        with self._lock:
            if len(self.hypotheses) >= self.max_hypotheses and hypothesis_id not in self.hypotheses:
                return

            timestamp = time.time()
            entry = {
                'id': hypothesis_id,
                'hypothesis_data': hypothesis_data,
                'evaluation_results': evaluation_results,
                'timestamp': timestamp,
                'step': step,
                'episode': episode,
                'training_context': training_context if training_context else {}
            }
            self.hypotheses[hypothesis_id] = entry
            self.metadata['total_evaluated'] = len(self.hypotheses)

            overall_score = evaluation_results.get('performance_score', 0.0)
            heapq.heappush(self.best_overall_heap, (-overall_score, hypothesis_id))
            if len(self.best_overall_heap) > self.top_n_to_keep:
                heapq.heappop(self.best_overall_heap)

            conservation_score = evaluation_results.get('conservation_score', 0.0)
            heapq.heappush(self.best_conservation_heap, (-conservation_score, hypothesis_id))
            if len(self.best_conservation_heap) > self.top_n_to_keep:
                heapq.heappop(self.best_conservation_heap)

            symmetry_score = evaluation_results.get('symmetry_score', 0.0)
            heapq.heappush(self.best_symmetry_heap, (-symmetry_score, hypothesis_id))
            if len(self.best_symmetry_heap) > self.top_n_to_keep:
                heapq.heappop(self.best_symmetry_heap)

            trajectory_fit_error = evaluation_results.get('trajectory_fit', float('inf'))
            heapq.heappush(self.best_trajectory_fit_heap, (trajectory_fit_error, hypothesis_id))
            if len(self.best_trajectory_fit_heap) > self.top_n_to_keep:
                heapq.heappop(self.best_trajectory_fit_heap)

            if self.metadata['total_evaluated'] - self.metadata.get('last_saved_count', 0) >= self.autosave_interval:
                self.save_state()

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.hypotheses.get(hypothesis_id)

    def search_hypotheses(self,
                          filters: Optional[Dict[str, Any]] = None,
                          sort_by: Optional[str] = None,
                          sort_ascending: bool = False,
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.hypotheses: return []

            results: List[Dict[str, Any]] = list(self.hypotheses.values())

            if filters:
                def check_filter(hyp: Dict[str,Any]) -> bool:
                    for key, value_cond in filters.items():
                        actual_value = hyp.get(key, hyp['evaluation_results'].get(key))
                        if actual_value is None: return False

                        if isinstance(value_cond, dict):
                            op = next(iter(value_cond))
                            val = value_cond[op]
                            if op == '__gt' and not (actual_value > val): return False
                            if op == '__lt' and not (actual_value < val): return False
                            if op == '__gte' and not (actual_value >= val): return False
                            if op == '__lte' and not (actual_value <= val): return False
                            if op == '__contains' and not (val in actual_value): return False
                        elif actual_value != value_cond:
                            return False
                    return True
                results = [h for h in results if check_filter(h)]

            if sort_by:
                def sort_key_func(hyp: Dict[str,Any]):
                    val = hyp.get(sort_by, hyp['evaluation_results'].get(sort_by))
                    if val is None:
                        return float('-inf') if not sort_ascending else float('inf')
                    return val
                try:
                    results.sort(key=sort_key_func, reverse=not sort_ascending)
                except TypeError as e:
                    print(f"Warning: Could not sort by '{sort_by}': {e}")

            if limit is not None:
                results = results[:limit]
            return results

    def get_best_hypothesis(self, criterion: str = 'overall') -> Optional[Dict[str, Any]]:
        with self._lock:
            heap_map = {
                'overall': self.best_overall_heap,
                'conservation': self.best_conservation_heap,
                'symmetry': self.best_symmetry_heap,
                'trajectory_fit': self.best_trajectory_fit_heap
            }
            target_heap = heap_map.get(criterion.lower())
            if not target_heap: return None

            if not target_heap: return None

            best_items = heapq.nsmallest(1, target_heap)
            if not best_items: return None

            best_id = best_items[0][1]
            return self.hypotheses.get(best_id)

    def get_top_n_hypotheses(self, criterion: str = 'overall', n: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            heap_map = {
                'overall': self.best_overall_heap,
                'conservation': self.best_conservation_heap,
                'symmetry': self.best_symmetry_heap,
                'trajectory_fit': self.best_trajectory_fit_heap
            }
            target_heap = heap_map.get(criterion.lower())
            if not target_heap: return []

            top_items = heapq.nsmallest(n, target_heap)
            top_ids = [item[1] for item in top_items]
            return [self.hypotheses[id_] for id_ in top_ids if id_ in self.hypotheses]

    def get_training_statistics(self) -> Dict[str, Any]:
        with self._lock:
            stats = self.metadata.copy()
            stats['current_time'] = time.time()
            stats['runtime_seconds'] = stats['current_time'] - stats['start_time']

            if self.hypotheses:
                perf_scores = [h['evaluation_results'].get('performance_score', 0)
                               for h in self.hypotheses.values()
                               if isinstance(h['evaluation_results'].get('performance_score'), (int, float))]
                stats['avg_performance_score'] = np.mean(perf_scores) if perf_scores else 0
                stats['median_performance_score'] = np.median(perf_scores) if perf_scores else 0
                stats['std_performance_score'] = np.std(perf_scores) if perf_scores else 0
            return stats

    def save_state(self, file_name_base: str = "hypothesis_tracker_state") -> None:
        with self._lock:
            self.metadata['last_saved_count'] = self.metadata['total_evaluated']
            self.metadata['last_save_time'] = time.time()

            hyp_path = os.path.join(self.save_directory, f"{file_name_base}_hypotheses.pkl")
            with open(hyp_path, 'wb') as f:
                pickle.dump(self.hypotheses, f)

            full_state_path = os.path.join(self.save_directory, f"{file_name_base}_full.pkl")
            full_state_content = {
                'metadata': self.metadata,
                'best_overall_heap': self.best_overall_heap,
                'best_conservation_heap': self.best_conservation_heap,
                'best_symmetry_heap': self.best_symmetry_heap,
                'best_trajectory_fit_heap': self.best_trajectory_fit_heap,
            }
            with open(full_state_path, 'wb') as f:
                pickle.dump(full_state_content, f)

    def load_state(self, file_name_base: str = "hypothesis_tracker_state") -> None:
        with self._lock:
            hyp_path = os.path.join(self.save_directory, f"{file_name_base}_hypotheses.pkl")
            full_state_path = os.path.join(self.save_directory, f"{file_name_base}_full.pkl")

            if os.path.exists(hyp_path) and os.path.exists(full_state_path):
                try:
                    with open(hyp_path, 'rb') as f:
                        self.hypotheses = pickle.load(f)

                    with open(full_state_path, 'rb') as f:
                        loaded_state = pickle.load(f)
                        self.metadata = loaded_state.get('metadata', self.metadata)
                        self.best_overall_heap = loaded_state.get('best_overall_heap', [])
                        self.best_conservation_heap = loaded_state.get('best_conservation_heap', [])
                        self.best_symmetry_heap = loaded_state.get('best_symmetry_heap', [])
                        self.best_trajectory_fit_heap = loaded_state.get('best_trajectory_fit_heap', [])

                    for h_heap in [self.best_overall_heap, self.best_conservation_heap, self.best_symmetry_heap, self.best_trajectory_fit_heap]:
                        heapq.heapify(h_heap)

                    self.metadata['total_evaluated'] = len(self.hypotheses)
                except Exception as e:
                    print(f"Error loading HypothesisTracker state: {e}. Starting fresh.")
                    self._reset_state_to_default()
            else:
                self._reset_state_to_default()

    def _reset_state_to_default(self):
        self.hypotheses = {}
        self.metadata = {'total_evaluated': 0, 'last_saved_count': 0, 'start_time': time.time()}
        self.best_overall_heap = []
        self.best_conservation_heap = []
        self.best_symmetry_heap = []
        self.best_trajectory_fit_heap = []

    def export_best_hypotheses(self, output_file_path: str, criterion: str = 'overall', n: int = 10, format: str = 'json'):
        top_n = self.get_top_n_hypotheses(criterion=criterion, n=n)
        if not top_n:
            return

        try:
            if os.path.dirname(output_file_path):
                 os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'w') as f:
                if format == 'json':
                    json.dump(top_n, f, indent=2)
                elif format == 'txt':
                    for i, hyp_data in enumerate(top_n):
                        f.write(f"--- Rank {i+1} ({criterion}) ---\nID: {hyp_data.get('id')}\n")
                        f.write(f"Hypothesis: {json.dumps(hyp_data.get('hypothesis_data'))}\n")
                        f.write(f"Evaluation: {json.dumps(hyp_data.get('evaluation_results'))}\n\n")
                else:
                    print(f"Unsupported export format: {format}")
        except Exception as e:
            print(f"Error exporting hypotheses: {e}")

    def clear_all_data(self):
        with self._lock:
            self._reset_state_to_default()

class JanusTrainingIntegration:
    def __init__(self, tracker: HypothesisTracker):
        self.tracker = tracker

    def on_hypothesis_evaluated(self,
                                hypothesis_data: HypothesisData,
                                evaluation_results: Dict[str, Any],
                                step: Optional[int] = None,
                                episode: Optional[int] = None,
                                training_context: Optional[Dict[str, Any]] = None
                                ) -> str:
        hypothesis_id = self.tracker._generate_id(hypothesis_data)
        self.tracker.add_hypothesis(
            hypothesis_id=hypothesis_id,
            hypothesis_data=hypothesis_data,
            evaluation_results=evaluation_results,
            step=step,
            episode=episode,
            training_context=training_context
        )
        return hypothesis_id

    def get_best_discovered_law(self, criterion: str = 'overall') -> Optional[Dict[str, Any]]:
        return self.tracker.get_best_hypothesis(criterion=criterion)

    def get_top_n_laws(self, n: int = 5, criterion: str = 'overall') -> List[Dict[str, Any]]:
        return self.tracker.get_top_n_hypotheses(criterion=criterion, n=n)

if __name__ == '__main__':
    tracker = HypothesisTracker(max_hypotheses=1000, autosave_interval=5, save_directory="test_hypothesis_data_main")
    tracker.clear_all_data()

    common_eval = {'performance_score':0.0,'conservation_score':0.0,'symmetry_score':0.0,'trajectory_fit':1.0,'functional_form':""}
    for i in range(15):
        hyp = {'form': f'x+{i}', 'params': {'c':i}}
        eval_r = common_eval.copy()
        eval_r.update({k:np.random.rand() for k in ['performance_score','conservation_score','symmetry_score']})
        eval_r['trajectory_fit'] = 1.0 / (eval_r['performance_score']+1e-3)
        eval_r['functional_form'] = hyp['form']
        tracker.add_hypothesis(tracker._generate_id(hyp), hyp, eval_r, i, i//10)

    print(f"Total tracked: {tracker.metadata['total_evaluated']}")
    for crit in ['overall', 'conservation', 'trajectory_fit']:
        best = tracker.get_best_hypothesis(criterion=crit)
        print(f"Best {crit}: {best['id'][:8] if best else 'N/A'}, Score: {best['evaluation_results'].get(crit+'_score', best['evaluation_results'].get('trajectory_fit', float('nan'))):.3f}" if best else "")

    tracker.save_state()
    del tracker
    loaded = HypothesisTracker(save_directory="test_hypothesis_data_main")
    print(f"Loaded: {loaded.metadata['total_evaluated']}")

    integration = JanusTrainingIntegration(loaded)
    best_law = integration.get_best_discovered_law()
    print(f"Best via integration: {best_law['id'][:8] if best_law else 'N/A'}")

    import shutil
    # Make sure this path is correct and intended for cleanup
    cleanup_paths = ["test_hypothesis_data_main", "example_training_run.jsonl"]
    # The log file "janus_training_log.jsonl" might be created by live_monitor test if run before this one
    # Add it to cleanup if it's created at root.
    if os.path.exists("janus_training_log.jsonl"):
        cleanup_paths.append("janus_training_log.jsonl")

    for p in cleanup_paths:
        if os.path.exists(p):
            if os.path.isdir(p): shutil.rmtree(p)
            else: os.remove(p)
    print("Cleanup done.")
