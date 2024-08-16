import json
import subprocess
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import psutil
from overseer.utils.logging import OverseerLogger
from overseer.config.config import OverseerConfig

class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class JobResources:
    cpus: int = 1
    memory_gb: int = 4
    time_limit_hours: int = 12
    gpu: bool = False

@dataclass
class JobMetrics:
    start_time: float
    end_time: float
    cpu_usage: float
    memory_usage: float
    exit_code: int

@dataclass
class JobInfo:
    id: str
    status: JobStatus
    submit_time: float
    resources: JobResources
    metrics: Optional[JobMetrics] = None

class ComputeEnvironment(ABC):
    @abstractmethod
    def submit_job(self, command: str, resources: JobResources) -> str:
        pass

    @abstractmethod
    def check_job_status(self, job_id: str) -> JobStatus:
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        pass

    @abstractmethod
    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_job_metrics(self, job_id: str) -> JobMetrics:
        pass

class LocalEnvironment(ComputeEnvironment):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def submit_job(self, job_id: str, command: str) -> None:
        self.logger.info(f"Submitting job {job_id} with command: {command}")
        
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            self.jobs[job_id] = {
                "process": process,
                "status": JobStatus.RUNNING,
                "start_time": time.time(),
                "end_time": None,
                "cpu_usage": [],
                "memory_usage": [],
            }
            
            self.logger.debug(f"Job {job_id} started with PID {process.pid}")
            
            # Start monitoring the process
            self._monitor_job(job_id)
        except Exception as e:
            self.logger.error(f"Error submitting job {job_id}: {str(e)}")
            self.jobs[job_id] = {
                "process": None,
                "status": JobStatus.FAILED,
                "start_time": time.time(),
                "end_time": time.time(),
                "cpu_usage": [],
                "memory_usage": [],
            }


    def _monitor_job(self, job_id: str) -> None:
        process = self.jobs[job_id]["process"]
        
        while process.poll() is None:
            try:
                proc = psutil.Process(process.pid)
                cpu_percent = proc.cpu_percent(interval=1)
                memory_percent = proc.memory_percent()
                
                self.jobs[job_id]["cpu_usage"].append(cpu_percent)
                self.jobs[job_id]["memory_usage"].append(memory_percent)
                
                self.logger.debug(f"Job {job_id} - CPU: {cpu_percent}%, Memory: {memory_percent}%")
                
                time.sleep(1)
            except psutil.NoSuchProcess:
                self.logger.warning(f"Process for job {job_id} no longer exists")
                break
            except Exception as e:
                self.logger.error(f"Error monitoring job {job_id}: {str(e)}")
        
        self._finalize_job(job_id)

    def _monitor_job(self, job_id: str) -> None:
        process = self.jobs[job_id]["process"]
        
        while process.poll() is None:
            try:
                proc = psutil.Process(process.pid)
                cpu_percent = proc.cpu_percent(interval=1)
                memory_percent = proc.memory_percent()
                
                self.jobs[job_id]["cpu_usage"].append(cpu_percent)
                self.jobs[job_id]["memory_usage"].append(memory_percent)
                
                self.logger.debug(f"Job {job_id} - CPU: {cpu_percent}%, Memory: {memory_percent}%")
                
                time.sleep(1)
            except psutil.NoSuchProcess:
                self.logger.warning(f"Process for job {job_id} no longer exists")
                break
            except Exception as e:
                self.logger.error(f"Error monitoring job {job_id}: {str(e)}")
        
        self._finalize_job(job_id)



    def _finalize_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        process = job["process"]
        
        job["end_time"] = time.time()
        job["status"] = JobStatus.COMPLETED if process.returncode == 0 else JobStatus.FAILED
        
        self.logger.info(f"Job {job_id} finished with status {job['status'].value}")
        self.logger.debug(f"Job {job_id} exit code: {process.returncode}")

    def get_job_status(self, job_id: str) -> JobStatus:
        self.logger.debug(f"Getting status for job {job_id}")
        return self.jobs[job_id]["status"]

    def get_job_metrics(self, job_id: str) -> JobMetrics:
        self.logger.debug(f"Getting metrics for job {job_id}")
        job = self.jobs[job_id]
        
        return JobMetrics(
            start_time=job["start_time"],
            end_time=job["end_time"] or time.time(),
            cpu_usage=sum(job["cpu_usage"]) / len(job["cpu_usage"]) if job["cpu_usage"] else 0,
            memory_usage=sum(job["memory_usage"]) / len(job["memory_usage"]) if job["memory_usage"] else 0,
            exit_code=job["process"].returncode if job["process"] else -1
        )   

class ComputeManager:
    def __init__(self, config: OverseerConfig):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.environment = self._get_compute_environment()
        self.sim_dir = Path(self.config.get('directories', {}).get('elmfire_sim_dir', ''))
        self.start_script = self.sim_dir / self.config.get('directories', {}).get('start_script', '01-run.sh')
        
        if not self.start_script.exists():
            raise FileNotFoundError(f"Start script not found: {self.start_script}")
        
        self.logger.info(f"Initialized ComputeManager with simulation directory: {self.sim_dir}")
        self.logger.info(f"Using start script: {self.start_script}")

    def _get_compute_environment(self) -> ComputeEnvironment:
        env_type = self.config.get('compute_environment', 'local')
        if env_type == 'local':
            return LocalEnvironment(self.logger)
        else:
            raise ValueError(f"Unsupported compute environment: {env_type}")

    def submit_simulation(self) -> str:
        command = f"bash {self.start_script.name}"
        resources = JobResources(**self.config.get('compute_resources', {}))
        job_id = self.environment.submit_job(command, resources, cwd=self.sim_dir)
        self.logger.info(f"Submitted simulation job {job_id}")
        return job_id

    def check_simulation_status(self, job_id: str) -> JobStatus:
        return self.environment.check_job_status(job_id)

    def cancel_simulation(self, job_id: str) -> bool:
        return self.environment.cancel_job(job_id)

    def retrieve_simulation_results(self, job_id: str) -> Dict[str, Any]:
        results = self.environment.retrieve_results(job_id)
        self.logger.info(f"Retrieved results for job {job_id}: {json.dumps(results, indent=2)}")
        return results

    def wait_for_simulation(self, job_id: str, check_interval: int = 60) -> JobStatus:
        while True:
            status = self.check_simulation_status(job_id)
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                self.logger.info(f"Simulation {job_id} finished with status: {status.value}")
                return status
            time.sleep(check_interval)

    def add_compute_resources(self, resources: JobResources):
        # This method would be implemented differently for different environments
        # For now, we'll just log the request
        self.logger.info(f"Request to add compute resources: {asdict(resources)}")
        # In a real implementation, you might update a resource pool or
        # communicate with a cluster management system


def main():
    # Initialize the configuration
    config_path = Path(__file__).parent.parent / 'config' / 'elmfire_config.yaml'
    config = OverseerConfig(config_path)

    # Initialize the ComputeManager
    try:
        compute_manager = ComputeManager(config)
    except FileNotFoundError as e:
        print(f"Error initializing ComputeManager: {e}")
        return

    # Submit a simulation
    job_id = compute_manager.submit_simulation()
    print(f"Submitted simulation job with ID: {job_id}")

    # Wait for the simulation to complete
    final_status = compute_manager.wait_for_simulation(job_id)
    print(f"Simulation completed with status: {final_status}")

    # Retrieve and check the results
    results = compute_manager.retrieve_simulation_results(job_id)
    if results:
        print("Simulation results:")
        print(json.dumps(results, indent=2))
        
        # Check if the job completed successfully
        if results.get('status') == JobStatus.COMPLETED.value:
            print("Simulation completed successfully.")
        else:
            print(f"Simulation did not complete successfully. Status: {results.get('status')}")
        
        # Check if metrics are available
        if 'metrics' in results:
            print("Job metrics:")
            print(json.dumps(results['metrics'], indent=2))
        else:
            print("No job metrics available.")
    else:
        print("No results retrieved for the simulation.")

if __name__ == "__main__":
    main()