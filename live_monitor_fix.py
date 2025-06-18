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
    """
    Logs training metrics to various backends like memory, file, or Redis.

    Attributes:
        backends (List[str]): A list of backend names to use for logging.
            Supported: "file", "memory", "redis".
        log_file_path (str): Path to the log file if "file" backend is used.
        max_memory_len (int): Maximum number of log entries to keep in memory
            if "memory" backend is used.
        memory_logs (deque): A deque to store logs in memory.
        file_handler (Optional[IO]): File handler for the log file.
        redis_client (Optional[redis.Redis]): Redis client instance.
        redis_channel (str): Redis channel to publish logs to.
    """
    _instances: weakref.WeakSet = weakref.WeakSet()

    def __init__(self,
                 backends: List[str] = ["file", "memory"],
                 log_file_path: str = "janus_training_log.jsonl",
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_channel: str = 'janus_training_metrics',
                 max_memory_len: int = 10000) -> None:
        """
        Initializes the TrainingLogger.

        Args:
            backends: List of backend names (e.g., ["file", "memory", "redis"]).
            log_file_path: Path for file logging.
            redis_host: Hostname for Redis server.
            redis_port: Port for Redis server.
            redis_channel: Redis channel for publishing metrics.
            max_memory_len: Max length for in-memory log deque.
        """
        self.backends: List[str] = [b.lower() for b in backends]
        self.log_file_path: str = log_file_path
        self.max_memory_len: int = max_memory_len

        self.memory_logs: deque[Dict[str, Any]] = deque(maxlen=self.max_memory_len)
        self.file_handler: Optional[Any] = None # Should be Optional[IO[str]] but Any for open()
        self.redis_client: Optional[redis.Redis] = None
        self.redis_channel: str = redis_channel

        if "file" in self.backends:
            self._setup_file_logging()

        if "redis" in self.backends:
            if not _REDIS_AVAILABLE:
                print("Warning: Redis backend requested but redis-py not installed. Skipping Redis.")
                if "redis" in self.backends: self.backends.remove("redis")
            else:
                self._setup_redis_logging(redis_host, redis_port)

        TrainingLogger._instances.add(self)

    def _setup_file_logging(self) -> None:
        """Sets up file logging by opening the log file."""
        try:
            # Ensure directory exists
            if os.path.dirname(self.log_file_path): # Create only if path has a directory part
                os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            # Open in append mode, create if not exists
            self.file_handler = open(self.log_file_path, 'a')
        except IOError as e:
            print(f"Error setting up file logging at {self.log_file_path}: {e}")
            if "file" in self.backends: self.backends.remove("file")

    def _setup_redis_logging(self, host: str, port: int) -> None:
        """
        Sets up Redis logging by connecting to the Redis server.

        Args:
            host: Redis server hostname.
            port: Redis server port.
        """
        try:
            self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
            self.redis_client.ping() # Check connection
            print(f"Successfully connected to Redis at {host}:{port}")
        except Exception as e: # Broad exception for redis connection errors
            print(f"Error setting up Redis logging (host={host}, port={port}): {e}")
            if "redis" in self.backends: self.backends.remove("redis")
            self.redis_client = None

    def log_metrics(self, **kwargs: Any) -> None:
        """
        Logs a dictionary of metrics to all configured backends.

        A 'timestamp' field is automatically added to the log entry.

        Args:
            **kwargs: Arbitrary key-value pairs representing the metrics to log.
                      Example: logger.log_metrics(reward=0.9, loss=0.1)
        """
        log_entry: Dict[str, Any] = {'timestamp': time.time()}
        log_entry.update(kwargs) # Add all provided metrics

        if "memory" in self.backends:
            self.memory_logs.append(log_entry)

        if "file" in self.backends and self.file_handler:
            try:
                json_entry: str = json.dumps(log_entry)
                self.file_handler.write(json_entry + '\n')
                self.file_handler.flush() # Ensure it's written immediately
            except Exception as e: # Broad exception for file write errors
                print(f"Error writing to log file: {e}")

        if "redis" in self.backends and self.redis_client:
            try:
                json_entry: str = json.dumps(log_entry)
                self.redis_client.publish(self.redis_channel, json_entry)
            except Exception as e: # Broad exception for Redis publish errors
                print(f"Error publishing to Redis: {e}")

    def get_memory_logs(self) -> List[Dict[str, Any]]:
        """
        Retrieves all logs currently stored in memory.

        Returns:
            A list of log entry dictionaries.
        """
        return list(self.memory_logs)

    def close(self) -> None:
        """
        Closes any open resources, such as file handlers.
        Should be called when logging is finished.
        """
        if self.file_handler:
            try:
                self.file_handler.close()
            except Exception as e: # Broad exception for file close errors
                print(f"Error closing log file: {e}")
            self.file_handler = None
        # Redis client from redis-py typically manages its connections automatically
        # and doesn't require an explicit close() unless using a connection pool
        # with a context manager.
        # print("TrainingLogger closed.")

    def __del__(self) -> None:
        """Ensures resources are closed when the object is garbage collected."""
        self.close()

    @classmethod
    def get_all_instances(cls) -> List['TrainingLogger']:
        """
        Returns a list of all active TrainingLogger instances.
        This is useful for accessing loggers globally, e.g., by LiveMonitor.

        Returns:
            A list of TrainingLogger instances.
        """
        return list(cls._instances)


class LiveMonitor:
    """
    Visualizes training metrics in real-time using Matplotlib.

    It can source data from memory (via a TrainingLogger instance), a log file,
    or a Redis pub/sub channel. The monitor can also accept direct log entries
    if configured with "callback" data source.

    Attributes:
        data_source_type (str): Source of the data ("memory", "file", "redis", "callback").
        log_file_path (str): Path to log file if data_source_type is "file".
        update_interval_sec (float): How often to refresh data and plot (in seconds).
        plot_config (Dict[str, Any]): Configuration for which metrics to plot and how.
        data_history (Dict[str, deque]): Stores recent data points for plotting.
            Keys are metric names, values are deques of metric values.
            Includes a 'timestamps' key. Max length of deques is 500.
        is_running (bool): Flag indicating if the monitoring loop is active.
        thread (Optional[threading.Thread]): Thread for the main data fetching loop.
        animation (Optional[FuncAnimation]): Matplotlib animation object.
        logger_instance (Optional[TrainingLogger]): Instance of TrainingLogger if
            data_source_type is "memory".
        redis_client (Optional[redis.Redis]): Redis client for "redis" data source.
        redis_channel (str): Redis channel for "redis" data source.
        redis_pubsub (Optional[redis.client.PubSub]): Redis pub/sub object.
        redis_thread (Optional[threading.Thread]): Thread for listening to Redis messages.
        fig (Optional[plt.Figure]): Matplotlib figure object.
        axes (Union[List[plt.Axes], plt.Axes, None]): Matplotlib axes object(s).
        lines (Dict[str, plt.Line2D]): Dictionary mapping metric keys to Matplotlib Line2D objects.
    """
    def __init__(self,
                 data_source: str = "memory", # "file", "redis", "memory", "callback"
                 log_file_path: str = "janus_training_log.jsonl",
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_channel: str = 'janus_training_metrics',
                 logger_instance: Optional[TrainingLogger] = None,
                 update_interval_sec: float = 1.0,
                 plot_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the LiveMonitor.

        Args:
            data_source: Specifies where to get data ("memory", "file", "redis", "callback").
            log_file_path: Path to the log file (used if data_source is "file").
            redis_host: Redis server hostname (used if data_source is "redis").
            redis_port: Redis server port (used if data_source is "redis").
            redis_channel: Redis channel to subscribe to (used if data_source is "redis").
            logger_instance: An instance of TrainingLogger (used if data_source is "memory").
            update_interval_sec: Interval in seconds for updating the plot.
            plot_config: A dictionary defining which metrics to plot and how.
                         If None, a default configuration is used.
                         Example:
                         {
                             'reward': {'label': 'Reward', 'color': 'blue', 'ax_idx': 0},
                             'loss': {'label': 'Loss', 'color': 'red', 'ax_idx': 1}
                         }

        Raises:
            ImportError: If 'redis' data_source is selected but redis-py is not installed.
        """

        if not _MPL_AVAILABLE:
            print("Matplotlib is not available. LiveMonitor cannot create plots. Only data retrieval will work.")

        self.data_source_type: str = data_source.lower()
        self.log_file_path: str = log_file_path
        self.update_interval_sec: float = update_interval_sec
        self.plot_config: Dict[str, Any] = plot_config if plot_config else self._default_plot_config()

        self.data_history: Dict[str, deque[Any]] = {key: deque(maxlen=500) for key in self.plot_config.keys()}
        self.data_history['timestamps'] = deque(maxlen=500) # Stores timestamps or step numbers

        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.animation: Optional[FuncAnimation] = None # type: ignore[name-defined] # FuncAnimation if _MPL_AVAILABLE

        self.logger_instance: Optional[TrainingLogger] = logger_instance
        self.redis_client: Optional[redis.Redis] = None
        self.redis_channel: str = redis_channel
        self.redis_pubsub: Optional[redis.client.PubSub] = None # type: ignore[name-defined] # PubSub if _REDIS_AVAILABLE
        self.redis_thread: Optional[threading.Thread] = None
        self._file_last_pos: int = 0
        self._memory_last_index: int = 0

        self.fig: Optional[plt.Figure] = None # type: ignore[name-defined] # Figure if _MPL_AVAILABLE
        self.axes: Union[List[plt.Axes], plt.Axes, None] = None # type: ignore[name-defined] # Axes if _MPL_AVAILABLE
        self.lines: Dict[str, plt.Line2D] = {} # type: ignore[name-defined] # Line2D if _MPL_AVAILABLE


        if self.data_source_type == "redis":
            if not _REDIS_AVAILABLE:
                raise ImportError("Redis backend selected for LiveMonitor, but redis-py is not installed.")
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)


        if _MPL_AVAILABLE:
            self._setup_plot()

    def _default_plot_config(self) -> Dict[str, Any]:
        """
        Provides a default configuration for plotting common training metrics.

        Returns:
            A dictionary defining plot configurations for various metrics.
        """
        return {
            'reward': {'label': 'Total Reward', 'color': 'blue', 'ax_idx': 0},
            'conservation_bonus': {'label': 'Conservation Bonus', 'color': 'green', 'ax_idx': 1},
            'symmetry_score': {'label': 'Symmetry Score', 'color': 'red', 'ax_idx': 1},
            'best_hypothesis_score': {'label': 'Best Hypothesis Score', 'color': 'purple', 'ax_idx': 0},
            'entropy_production': {'label': 'Entropy Production', 'color': 'orange', 'ax_idx': 2},
            'step': {'label': 'Training Step', 'color': 'grey', 'ax_idx': 3, 'plot_type': 'line'} # 'step' is often x-axis
        }

    def _setup_plot(self) -> None:
        """
        Initializes the Matplotlib figure and axes for plotting based on plot_config.
        This method is called only if Matplotlib is available.
        """
        if not _MPL_AVAILABLE: return

        num_subplots: int = max(pc.get('ax_idx', 0) for pc in self.plot_config.values()) + 1
        self.fig, self.axes = plt.subplots(num_subplots, 1, figsize=(10, 2 * num_subplots), sharex=True) # type: ignore[assignment]
        if num_subplots == 1: self.axes = [self.axes] # type: ignore[list-item]

        self.lines = {}
        for key, config in self.plot_config.items():
            if key == 'step': continue # 'step' is usually the x-axis, not plotted as a line itself
            ax_idx = config.get('ax_idx', 0)
            if not (0 <= ax_idx < num_subplots):
                print(f"Warning: Invalid ax_idx {ax_idx} for metric '{key}'. Defaulting to 0.")
                ax_idx = 0
            ax: plt.Axes = self.axes[ax_idx] # type: ignore[index]
            line, = ax.plot([], [], label=config.get('label', key), color=config.get('color', 'blue')) # type: ignore[call-overload]
            self.lines[key] = line
            ax.legend(loc='upper left')
            ax.set_ylabel(config.get('label', key))

        if self.axes:
            self.axes[-1].set_xlabel("Time / Steps") # type: ignore[index]
        if self.fig:
            self.fig.tight_layout()
        plt.ion() # type: ignore[attr-defined]

    def _update_plot(self, frame: Optional[Any] = None) -> bool:
        """
        Updates the plot with the latest data from data_history.
        This method is called by Matplotlib's FuncAnimation.

        Args:
            frame: Frame number (unused, but required by FuncAnimation).

        Returns:
            True, to continue the animation.
        """
        if not _MPL_AVAILABLE or not self.fig or not self.axes or not self.lines:
            return True # Continue animation, but do nothing if MPL not set up

        # Determine x-axis: prefer 'step' if available, else 'timestamps'
        x_data_key: str = 'step' if 'step' in self.data_history and len(self.data_history['step']) > 0 else 'timestamps'
        x_data: deque[Any] = self.data_history[x_data_key]

        if not list(x_data): # No data to plot
            return True

        for key, line in self.lines.items():
            if key in self.data_history and len(self.data_history[key]) > 0:
                y_data: List[Any] = list(self.data_history[key])
                current_x_data_list: List[Any] = list(x_data)

                # Align y_data with the most recent x_data points
                # This handles cases where some metrics might be logged less frequently
                plot_x_data: List[Any]
                if len(y_data) <= len(current_x_data_list):
                    plot_x_data = current_x_data_list[-len(y_data):]
                else: # Should not happen if data is processed correctly
                    plot_x_data = current_x_data_list
                    y_data = y_data[-len(current_x_data_list):]

                line.set_data(plot_x_data, y_data)

                ax_idx: int = self.plot_config[key].get('ax_idx', 0)
                ax: plt.Axes = self.axes[ax_idx] # type: ignore[index]
                ax.relim()
                ax.autoscale_view()

        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e: # Handle cases where the figure might be closed
            print(f"Error updating plot: {e}")
            # Potentially stop animation or mark as not running
        return True

    def _process_log_entry(self, entry: Dict[str, Any]) -> None:
        """
        Processes a single log entry and appends its data to data_history.

        Args:
            entry: A dictionary representing a log entry.
                   Must contain a 'timestamp' or other keys defined in plot_config.
        """
        timestamp: float = entry.get('timestamp', time.time())
        self.data_history['timestamps'].append(timestamp)

        for key in self.plot_config.keys():
            if key in entry:
                self.data_history[key].append(entry[key])
            # else:
                # Optionally, append a NaN or None if a key is missing, to keep lengths consistent
                # For now, we handle variable lengths in _update_plot
                # self.data_history[key].append(None)


    def _fetch_data_memory(self) -> None:
        """Fetches new log data from the associated TrainingLogger's memory logs."""
        logger_to_use: Optional[TrainingLogger] = self.logger_instance
        if not logger_to_use:
            # Try to find a global instance if one was not provided
            if TrainingLogger._instances:
                logger_to_use = next(iter(TrainingLogger._instances), None)

        if not logger_to_use:
            # print("LiveMonitor: No TrainingLogger instance found for memory data source.")
            return

        current_logs: deque[Dict[str, Any]] = logger_to_use.memory_logs
        current_logs_len: int = len(current_logs)

        # Basic reset/catch-up logic if memory logs were cleared or logger restarted
        if self._memory_last_index > current_logs_len:
            print("LiveMonitor: Detected reset or change in memory logs. Clearing local history.")
            self._memory_last_index = 0
            for key_deque in self.data_history.values():
                key_deque.clear()

        new_entries_count = 0
        for i in range(self._memory_last_index, current_logs_len):
            log_entry: Dict[str, Any] = current_logs[i]
            self._process_log_entry(log_entry)
            new_entries_count +=1

        self._memory_last_index = current_logs_len
        # if new_entries_count > 0: print(f"Fetched {new_entries_count} new entries from memory.")


    def _fetch_data_file(self) -> None:
        """Fetches new log data from the configured log file."""
        try:
            if not os.path.exists(self.log_file_path):
                # print(f"LiveMonitor: Log file {self.log_file_path} not found.")
                return
            with open(self.log_file_path, 'r') as f:
                f.seek(self._file_last_pos)
                new_lines = 0
                for line in f:
                    if not line.strip(): continue
                    try:
                        entry: Dict[str, Any] = json.loads(line)
                        self._process_log_entry(entry)
                        new_lines+=1
                    except json.JSONDecodeError:
                        print(f"Warning: LiveMonitor could not decode JSON line: {line.strip()}")
                self._file_last_pos = f.tell()
                # if new_lines > 0: print(f"Fetched {new_lines} new entries from file.")
        except IOError as e:
            print(f"Error reading log file {self.log_file_path}: {e}")
        except Exception as e: # Catch other potential errors
            print(f"Unexpected error in _fetch_data_file: {e}")


    def _listen_to_redis(self) -> None:
        """Listens for messages on a Redis channel and processes them as log entries."""
        if not self.redis_client or not _REDIS_AVAILABLE:
            print("LiveMonitor: Redis client not available for listening.")
            return

        self.redis_pubsub = self.redis_client.pubsub()
        try:
            self.redis_pubsub.subscribe(self.redis_channel)
            print(f"LiveMonitor: Subscribed to Redis channel '{self.redis_channel}'")
            for message in self.redis_pubsub.listen(): # type: ignore[union-attr] # listen() is on PubSub
                if not self.is_running:
                    break
                if message and message['type'] == 'message':
                    try:
                        # Ensure message['data'] is decoded if necessary (depends on redis client config)
                        data_str: str = message['data']
                        if isinstance(data_str, bytes): # Should be str due to decode_responses=True
                            data_str = data_str.decode('utf-8')
                        entry: Dict[str, Any] = json.loads(data_str)
                        self._process_log_entry(entry)
                    except json.JSONDecodeError:
                        print(f"Warning: LiveMonitor could not decode JSON from Redis: {message['data']}")
                    except Exception as e:
                        print(f"Error processing Redis message: {e}")
        except redis.exceptions.ConnectionError as e:
            print(f"LiveMonitor: Redis connection error: {e}. Stopping Redis listener.")
        except Exception as e:
            print(f"LiveMonitor: Unexpected error in Redis listener: {e}")
        finally:
            if self.redis_pubsub:
                try:
                    self.redis_pubsub.unsubscribe(self.redis_channel) # type: ignore[union-attr]
                    self.redis_pubsub.close() # type: ignore[union-attr]
                except Exception as e:
                    print(f"Error unsubscribing/closing Redis pubsub: {e}")
            print("LiveMonitor: Redis listener thread stopped.")


    def _data_loop(self) -> None:
        """
        Main loop for fetching data periodically based on the data_source_type.
        This loop runs in a separate thread.
        """
        print(f"LiveMonitor: Data loop started for source '{self.data_source_type}'.")
        while self.is_running:
            start_time = time.time()
            if self.data_source_type == "memory":
                self._fetch_data_memory()
            elif self.data_source_type == "file":
                self._fetch_data_file()
            elif self.data_source_type == "callback":
                # Data is pushed via direct_log_entry, so nothing to fetch here.
                pass
            # For "redis", data is fetched in its own thread (_listen_to_redis)

            # Sleep to maintain the update interval
            elapsed_time = time.time() - start_time
            sleep_duration = self.update_interval_sec - elapsed_time
            if sleep_duration > 0:
                time.sleep(sleep_duration)
        print("LiveMonitor: Data loop stopped.")


    def start_monitoring(self) -> None:
        """
        Starts the monitoring process.

        This includes initializing the plot, starting data fetching threads,
        and showing the Matplotlib window if GUI is available.
        """
        if self.is_running:
            print("LiveMonitor: Monitor is already running.")
            return

        self.is_running = True
        print(f"LiveMonitor: Starting monitoring (source: {self.data_source_type})...")

        # Reset data history if restarting
        for key_deque in self.data_history.values():
            key_deque.clear()
        self._file_last_pos = 0
        self._memory_last_index = 0


        if self.data_source_type == "redis":
            if not self.redis_client or not _REDIS_AVAILABLE:
                print("Error: LiveMonitor Redis client not initialized for Redis data source.")
                self.is_running = False
                return
            self.redis_thread = threading.Thread(target=self._listen_to_redis, daemon=True)
            self.redis_thread.start()

        # Data loop for memory, file, callback (Redis has its own listener thread)
        if self.data_source_type in ["memory", "file", "callback"]:
            self.thread = threading.Thread(target=self._data_loop, daemon=True)
            self.thread.start()

        if _MPL_AVAILABLE and plt.get_backend() and self.fig: # Ensure fig is created
            try:
                # The FuncAnimation should be stored as an instance variable
                # to prevent it from being garbage collected.
                self.animation = FuncAnimation(self.fig, self._update_plot, # type: ignore[assignment]
                                               interval=self.update_interval_sec * 1000,
                                               cache_frame_data=False) # Important for live data
                plt.show(block=False) # type: ignore[attr-defined]
                print("LiveMonitor: Matplotlib animation started.")
            except Exception as e: # Catch specific exceptions if possible
                print(f"LiveMonitor: Failed to start Matplotlib animation: {e}")
        elif _MPL_AVAILABLE and not self.fig:
             print("LiveMonitor: Matplotlib available, but plot setup failed or was skipped.")


    def stop_monitoring(self) -> None:
        """
        Stops the monitoring process.

        This includes stopping data fetching threads and closing the Matplotlib window.
        """
        if not self.is_running:
            print("LiveMonitor: Monitor is not running.")
            return

        print("LiveMonitor: Stopping monitoring...")
        self.is_running = False # Signal all threads to stop

        # Stop the main data loop thread (for memory, file, callback)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=max(1.0, self.update_interval_sec * 2))
            if self.thread.is_alive():
                 print("LiveMonitor: Data loop thread did not terminate cleanly.")

        # Stop the Redis listener thread
        if self.redis_thread and self.redis_thread.is_alive():
            # The _listen_to_redis loop checks self.is_running.
            # If it's blocked on pubsub.listen(), closing pubsub might be needed.
            if self.redis_pubsub:
                try:
                    self.redis_pubsub.unsubscribe(self.redis_channel) # type: ignore[union-attr]
                    self.redis_pubsub.close() # type: ignore[union-attr] # This should help break the listen() loop.
                except Exception as e:
                    print(f"Error closing Redis pubsub during stop: {e}")
            self.redis_thread.join(timeout=2.0) # Wait for thread to finish
            if self.redis_thread.is_alive():
                print("LiveMonitor: Redis listener thread did not terminate cleanly.")


        # Close Matplotlib figure
        if _MPL_AVAILABLE and hasattr(self, 'fig') and self.fig:
            try:
                plt.close(self.fig) # type: ignore[attr-defined]
            except Exception as e:
                print(f"Error closing Matplotlib figure: {e}")
        self.animation = None # Clear animation object

        print("LiveMonitor: Monitoring stopped.")


    def direct_log_entry(self, entry: Dict[str, Any]) -> None:
        """
        Allows direct injection of log entries into the monitor.
        This is primarily intended for use when data_source_type is "callback".

        Args:
            entry: A dictionary representing the log entry.
        """
        if self.data_source_type != "callback":
            print(f"Warning: direct_log_entry called but LiveMonitor source is '{self.data_source_type}', not 'callback'.")
        self._process_log_entry(entry)


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
