# live_monitor_fix.py
import json
import time
import threading
from collections import deque
from typing import List, Dict, Any, Optional, Union
import os
import weakref
import numpy as np # Added for example usage

# Attempt to import plotting libraries, but make them optional
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    # print("Warning: Matplotlib not found. Live plotting will be disabled.")

# Attempt to import Redis client, make it optional
try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    # print("Warning: Redis-py not found. Redis backend for logging will be disabled.")


class TrainingLogger:
    _instances = weakref.WeakSet()

    def __init__(self,
                 backends: List[str] = ["file", "memory"],
                 log_file_path: str = "janus_training_log.jsonl",
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_channel: str = 'janus_training_metrics',
                 max_memory_len: int = 10000):

        self.backends = [b.lower() for b in backends]
        self.log_file_path = log_file_path
        self.max_memory_len = max_memory_len

        self.memory_logs: deque = deque(maxlen=self.max_memory_len)
        self.file_handler: Optional[Any] = None
        self.redis_client: Optional[Any] = None
        self.redis_channel = redis_channel

        if "file" in self.backends:
            self._setup_file_logging()
        
        if "redis" in self.backends:
            if not _REDIS_AVAILABLE:
                print("Warning: Redis backend requested but redis-py not installed. Skipping Redis.")
                self.backends.remove("redis")
            else:
                self._setup_redis_logging(redis_host, redis_port)
        
        TrainingLogger._instances.add(self)

    def _setup_file_logging(self):
        try:
            # Ensure directory exists
            if os.path.dirname(self.log_file_path): # Create only if path has a directory part
                os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            # Open in append mode, create if not exists
            self.file_handler = open(self.log_file_path, 'a')
        except IOError as e:
            print(f"Error setting up file logging at {self.log_file_path}: {e}")
            if "file" in self.backends: self.backends.remove("file")

    def _setup_redis_logging(self, host: str, port: int):
        try:
            self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
            self.redis_client.ping() # Check connection
            print(f"Successfully connected to Redis at {host}:{port}")
        except Exception as e:
            print(f"Error setting up Redis logging (host={host}, port={port}): {e}")
            if "redis" in self.backends: self.backends.remove("redis")
            self.redis_client = None

    def log_metrics(self, **kwargs: Any):
        log_entry = {'timestamp': time.time()}
        log_entry.update(kwargs) # Add all provided metrics

        if "memory" in self.backends:
            self.memory_logs.append(log_entry)

        if "file" in self.backends and self.file_handler:
            try:
                json_entry = json.dumps(log_entry)
                self.file_handler.write(json_entry + '\n')
                self.file_handler.flush() # Ensure it's written immediately
            except Exception as e:
                print(f"Error writing to log file: {e}")

        if "redis" in self.backends and self.redis_client:
            try:
                json_entry = json.dumps(log_entry)
                self.redis_client.publish(self.redis_channel, json_entry)
            except Exception as e:
                print(f"Error publishing to Redis: {e}")
    
    def get_memory_logs(self) -> List[Dict[str, Any]]:
        return list(self.memory_logs)

    def close(self):
        if self.file_handler:
            try:
                self.file_handler.close()
            except Exception as e:
                print(f"Error closing log file: {e}")
            self.file_handler = None
        # Redis client typically doesn't need explicit close unless using connection pool with context manager
        # print("TrainingLogger closed.")

    def __del__(self):
        self.close()
    
    @classmethod
    def get_all_instances(cls):
        return list(cls._instances)


class LiveMonitor:
    def __init__(self,
                 data_source: str = "memory", # "file", "redis", "memory", "callback"
                 log_file_path: str = "janus_training_log.jsonl",
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_channel: str = 'janus_training_metrics',
                 logger_instance: Optional[TrainingLogger] = None,
                 update_interval_sec: float = 1.0,
                 plot_config: Optional[Dict[str, Any]] = None):

        if not _MPL_AVAILABLE:
            print("Matplotlib is not available. LiveMonitor cannot create plots. Only data retrieval will work.")

        self.data_source_type = data_source.lower()
        self.log_file_path = log_file_path
        self.update_interval_sec = update_interval_sec
        self.plot_config = plot_config if plot_config else self._default_plot_config()

        self.data_history: Dict[str, deque] = {key: deque(maxlen=500) for key in self.plot_config.keys()}
        self.data_history['timestamps'] = deque(maxlen=500)
        self.phase_transition_history: deque = deque(maxlen=500) # Original deque for raw events, might be removed later if not used
        self.data_history['phase_numeric'] = deque(maxlen=500)
        self.data_history['phase_timestamps'] = deque(maxlen=500)

        # Phase visualization attributes
        self.phase_to_numeric: Dict[str, int] = {
            'stagnation': 0,
            'exploration': 1,
            'breakthrough': 2,
            'simplification': 3,
            'refinement': 4
        }
        self.phase_colors: Dict[str, str] = {
            'stagnation': 'gray',
            'exploration': 'blue',
            'breakthrough': 'green',
            'simplification': 'orange',
            'refinement': 'purple'
        }

        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.animation: Optional[FuncAnimation] = None

        self.logger_instance = logger_instance
        self.redis_pubsub: Optional[Any] = None
        self.redis_thread: Optional[threading.Thread] = None
        self._file_last_pos = 0

        if self.data_source_type == "redis":
            if not _REDIS_AVAILABLE:
                raise ImportError("Redis backend selected for LiveMonitor, but redis-py is not installed.")
            self.redis_client = redis.Redis(host=redis_host, port=redis_port)
            self.redis_channel = redis_channel

        if _MPL_AVAILABLE:
            self._setup_plot()

    def _default_plot_config(self) -> Dict[str, Any]:
        return {
            'reward': {'label': 'Total Reward', 'color': 'blue', 'ax_idx': 0},
            'conservation_bonus': {'label': 'Conservation Bonus', 'color': 'green', 'ax_idx': 1},
            'symmetry_score': {'label': 'Symmetry Score', 'color': 'red', 'ax_idx': 1},
            'best_hypothesis_score': {'label': 'Best Hypothesis Score', 'color': 'purple', 'ax_idx': 0},
            'entropy_production': {'label': 'Entropy Production', 'color': 'orange', 'ax_idx': 2},
            'step': {'label': 'Training Step', 'color': 'grey', 'ax_idx': 3, 'plot_type': 'line'}, # Note: ax_idx 3 was already used by 'step'
            'phase_transitions': {
                'label': 'Training Phase',
                'color': 'purple', # Default, can be overridden by self.phase_colors
                'ax_idx': 3, # Will share with 'step' or use a new one if step is moved/changed
                'plot_type': 'step' # 'step' plot type is suitable for discrete phase changes
            }
        }

    def _setup_plot(self):
        if not _MPL_AVAILABLE: return

        num_subplots = max(pc['ax_idx'] for pc in self.plot_config.values()) + 1
        self.fig, self.axes = plt.subplots(num_subplots, 1, figsize=(10, 2 * num_subplots), sharex=True)
        if num_subplots == 1: self.axes = [self.axes]

        self.lines = {}
        for key, config in self.plot_config.items():
            if key == 'step': continue

            ax_idx = config['ax_idx']
            # Ensure axes list is long enough (it should be due to num_subplots calculation)
            if ax_idx >= len(self.axes):
                print(f"Warning: ax_idx {ax_idx} for '{key}' is out of bounds. Number of axes: {len(self.axes)}")
                # Potentially skip this plot or handle error, for now, let's assume it's correct
                # or that 'step' plot_type might not need a line in the same way
                if config.get('plot_type') != 'step' or key == 'phase_transitions': # phase_transitions needs a line
                     continue


            ax = self.axes[ax_idx]

            if key == 'phase_transitions':
                # Set y-axis ticks and labels for phase transitions
                ax.set_yticks(list(self.phase_to_numeric.values()))
                ax.set_yticklabels(list(self.phase_to_numeric.keys()))
                # Use a step plot for phases
                line, = ax.step([], [], where='post', label=config['label'], color=config.get('color', 'purple'))
                ax.set_ylabel(config['label']) # Set Y-label specifically for this axis
            else:
                line, = ax.plot([], [], label=config['label'], color=config['color'])
                ax.set_ylabel(config['label'])

            self.lines[key] = line
            ax.legend(loc='upper left')


        self.axes[-1].set_xlabel("Time / Steps")
        self.fig.tight_layout()
        plt.ion()

    def _update_plot(self, frame: Optional[Any] = None):
        if not _MPL_AVAILABLE: return True

        x_data_key_primary = 'step' if 'step' in self.data_history and len(self.data_history['step']) > 0 else 'timestamps'
        x_axis_primary = self.data_history[x_data_key_primary]

        # It's possible x_axis_primary is empty if no data has arrived for 'step' or 'timestamps'
        # We should still attempt to plot phases if they have their own timestamps.
        # However, if sharex is True, an empty primary x-axis might lead to issues.
        # For now, let's proceed with the logic that plots may proceed if they have data.

        for key, line in self.lines.items():
            config = self.plot_config[key]
            ax = self.axes[config['ax_idx']]
            current_y_data = None
            current_x_data_for_line = None

            if key == 'phase_transitions':
                if self.data_history['phase_timestamps'] and self.data_history['phase_numeric']:
                    current_x_data_for_line = list(self.data_history['phase_timestamps'])
                    current_y_data = list(self.data_history['phase_numeric'])
            elif key in self.data_history and self.data_history[key]: # For other plot types
                current_y_data = list(self.data_history[key])
                primary_x_list = list(x_axis_primary) # Use the determined primary x-axis

                if not primary_x_list and current_y_data: # If no primary x-axis data, cannot plot regular lines
                    current_x_data_for_line = None
                    current_y_data = None
                elif primary_x_list: # Only proceed if primary x-axis has data
                    if len(current_y_data) <= len(primary_x_list):
                        current_x_data_for_line = primary_x_list[-len(current_y_data):]
                    else:
                        # y_data is longer than primary x_axis, truncate y_data to match x_axis length
                        current_x_data_for_line = primary_x_list
                        current_y_data = current_y_data[:len(primary_x_list)]

            if current_x_data_for_line and current_y_data:
                line.set_data(current_x_data_for_line, current_y_data)
                ax.relim()
                ax.autoscale_view()
            elif key != 'phase_transitions': # Clear non-phase lines if no data
                # This handles the case where a metric stops reporting or has no initial data
                # We might not want to clear phase_transitions if it's sparse and meant to persist
                line.set_data([],[])
                ax.relim()
                ax.autoscale_view()

        if self.fig.canvas.manager and self.fig.canvas.manager.window: # Check if figure is valid
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        return True

    def _process_log_entry(self, entry: Dict[str, Any]):
        timestamp = entry.get('timestamp', time.time())
        self.data_history['timestamps'].append(timestamp)

        for key in self.plot_config.keys():
            if key in entry:
                self.data_history[key].append(entry[key])

    def _fetch_data_memory(self):
        logger_to_use = self.logger_instance
        if not logger_to_use:
            if TrainingLogger._instances:
                logger_to_use = next(iter(TrainingLogger._instances), None)

        if logger_to_use:
            all_logs = logger_to_use.get_memory_logs()
            if all_logs:
                temp_data_store = {k:[] for k in self.data_history.keys()}
                for log_entry in logger_to_use.memory_logs:
                    temp_data_store['timestamps'].append(log_entry.get('timestamp', time.time()))
                    for plot_key in self.plot_config.keys():
                         if plot_key in log_entry:
                            temp_data_store[plot_key].append(log_entry[plot_key])

                for k, v_list in temp_data_store.items():
                    self.data_history[k].clear()
                    self.data_history[k].extend(v_list)

    def _fetch_data_file(self):
        try:
            if not os.path.exists(self.log_file_path): return
            with open(self.log_file_path, 'r') as f:
                f.seek(self._file_last_pos)
                for line_content in f: # renamed 'line' to 'line_content'
                    try:
                        entry = json.loads(line_content)
                        self._process_log_entry(entry)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON line: {line_content.strip()}")
                self._file_last_pos = f.tell()
        except Exception as e:
            print(f"Error reading log file {self.log_file_path}: {e}")

    def _listen_to_redis(self):
        if not self.redis_client : return
        self.redis_pubsub = self.redis_client.pubsub()
        self.redis_pubsub.subscribe(self.redis_channel)
        print(f"LiveMonitor: Subscribed to Redis channel '{self.redis_channel}'")
        for message in self.redis_pubsub.listen():
            if not self.is_running: break
            if message['type'] == 'message':
                try:
                    entry = json.loads(message['data'])
                    self._process_log_entry(entry)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from Redis: {message['data']}")
                except Exception as e:
                    print(f"Error processing Redis message: {e}")
        print("LiveMonitor: Redis listener thread stopped.")

    def _data_loop(self):
        while self.is_running:
            if self.data_source_type == "memory":
                self._fetch_data_memory()
            elif self.data_source_type == "file":
                self._fetch_data_file()
            elif self.data_source_type == "callback":
                pass
            time.sleep(self.update_interval_sec)
        print("LiveMonitor: Data loop stopped.")

    def start_monitoring(self):
        if self.is_running:
            print("Monitor is already running.")
            return

        self.is_running = True
        print(f"LiveMonitor: Starting monitoring (source: {self.data_source_type})...")

        if self.data_source_type == "redis":
            if not self.redis_client:
                print("Error: Redis client not initialized for Redis data source.")
                self.is_running = False
                return
            self.redis_thread = threading.Thread(target=self._listen_to_redis, daemon=True)
            self.redis_thread.start()

        self.thread = threading.Thread(target=self._data_loop, daemon=True)
        self.thread.start()

        if _MPL_AVAILABLE and plt.get_backend():
            try:
                self.animation = FuncAnimation(self.fig, self._update_plot,
                                               interval=self.update_interval_sec * 1000, cache_frame_data=False)
                plt.show(block=False)
                print("LiveMonitor: Matplotlib animation started.")
            except Exception as e:
                print(f"LiveMonitor: Failed to start Matplotlib animation: {e}")

    def stop_monitoring(self):
        if not self.is_running:
            print("Monitor is not running.")
            return

        print("LiveMonitor: Stopping monitoring...")
        self.is_running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=self.update_interval_sec * 2)
        
        if self.redis_pubsub:
            try:
                self.redis_pubsub.unsubscribe(self.redis_channel)
                self.redis_pubsub.close()
            except Exception as e: print(f"Error closing redis pubsub: {e}")
            self.redis_pubsub = None
        
        if self.redis_thread and self.redis_thread.is_alive():
             self.redis_thread.join(timeout=2)
        
        if _MPL_AVAILABLE and hasattr(self, 'fig') and self.fig:
            try:
                plt.close(self.fig)
            except Exception as e: print(f"Error closing matplotlib figure: {e}")
        
        print("LiveMonitor: Monitoring stopped.")

    def direct_log_entry(self, entry: Dict[str, Any]):
        if self.data_source_type != "callback":
            print("Warning: direct_log_entry called but source is not 'callback'.")
        self._process_log_entry(entry)

    def log_phase_transition(self, event: Dict[str, Any]):
        """Logs a phase transition event."""
        # self.phase_transition_history.append(event) # Keep original history if needed for other purposes

        timestamp = event.get('timestamp', time.time()) # Ensure timestamp exists
        phase_type = event.get('type')

        if phase_type is not None:
            self.data_history['phase_timestamps'].append(timestamp)
            numeric_phase = self.phase_to_numeric.get(phase_type, -1) # Default to -1 for unknown
            self.data_history['phase_numeric'].append(numeric_phase)
        else:
            print(f"Warning: Phase transition event missing 'type': {event}")


if __name__ == '__main__':
    print("Starting Live Monitor Example...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "example_training_run.jsonl")

    if _REDIS_AVAILABLE:
         logger = TrainingLogger(backends=["redis", "memory"], redis_channel="janus_test_metrics")
    else:
         logger = TrainingLogger(backends=["file", "memory"], log_file_path=log_file)

    if "redis" in logger.backends:
        monitor = LiveMonitor(data_source="redis", redis_channel=logger.redis_channel, update_interval_sec=0.5)
    else:
        monitor = LiveMonitor(data_source="file", log_file_path=log_file, update_interval_sec=0.5)

    monitor.start_monitoring()

    print(f"Simulating training loop for 20 seconds. Log file: {log_file if 'file' in logger.backends else 'N/A'}")
    try:
        for i in range(40):
            reward = np.sin(i / 10) + np.random.rand() * 0.1
            conservation = np.exp(- (i % 10) / 5) + np.random.rand() * 0.05
            symmetry = (np.cos(i / 5) + 1) / 2 + np.random.rand() * 0.05
            best_score = reward + conservation + symmetry + np.random.rand() * 0.2
            entropy = np.random.rand() * 0.01 if i%2==0 else 0.02

            logger.log_metrics(
                step=i, episode=(i // 10), reward=float(reward),
                conservation_bonus=float(conservation), symmetry_score=float(symmetry),
                discovered_law_params={'p1': float(np.random.rand()), 'p2': str(np.random.choice(['a','b']))},
                best_hypothesis_score=float(best_score), entropy_production=entropy )

            # Simulate phase transitions for testing phase transition plotting
            current_time = time.time() # Use a consistent timestamp for events in the same iteration
            if i == 5:
                monitor.log_phase_transition({'timestamp': current_time, 'type': 'exploration', 'step': i})
            elif i == 15:
                monitor.log_phase_transition({'timestamp': current_time, 'type': 'breakthrough', 'step': i})
            elif i == 25:
                # Using 'simplification' as an example, assuming it's in phase_to_numeric
                # Or stick to 'refinement' if 'simplification' might not always be there.
                # Based on previous context, 'refinement' is present.
                monitor.log_phase_transition({'timestamp': current_time, 'type': 'refinement', 'step': i})
            elif i == 35:
                monitor.log_phase_transition({'timestamp': current_time, 'type': 'stagnation', 'step': i})

            if _MPL_AVAILABLE and hasattr(monitor,'fig') and monitor.fig.canvas.manager :
                plt.pause(0.01)
            time.sleep(0.5)
    except KeyboardInterrupt: print("Simulation interrupted.")
    finally:
        print("Simulation finished.")
        logger.close()
        monitor.stop_monitoring()
        for inst in TrainingLogger.get_all_instances(): inst.close()
        if _MPL_AVAILABLE and plt.get_fignums(): plt.show(block=True)
        print("Exiting example.")
