import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Optional

from models.api_models import RunInfo

logger = logging.getLogger("api_state")


class ApiState:
    """ API状态管理"""

    def __init__(self):
        self.lock = threading.RLock()
        self.agent_data: Dict[str, Dict] = {}
        self.runs: Dict[str, RunInfo] = {}
        self.current_run_id = None
        self._executor = ThreadPoolExecutor(max_workers=5)
        self._analysis_tasks: Dict[str, Future] = {}  # 跟踪分析任务

    @property
    def current_run_id(self) -> Optional[str]:
        """获取当前正在运行的任务ID"""
        with self.lock:
            return self.current_run_id

    @current_run_id.setter
    def current_run_id(self, run_id: str):
        """设置当前正在运行任务的ID"""
        with self.lock:
            self.current_run_id = run_id


