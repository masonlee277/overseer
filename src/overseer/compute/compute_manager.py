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
    def __init__(self, logger: OverseerLogger):
        self.jobs: Dict[str, JobInfo] = {}
        self.logger = logger

    def submit_job(self, command: str, resources: JobResources, cwd: Path) -> str:
        job_id = str(uuid.uuid4())
        self.logger.info(f"Submitting local job {job_id}")
        
        def run_job():
            job_info = JobInfo(id=job_id, status=JobStatus.RUNNING, submit_time=time.time(), resources=resources)
            self.jobs[job_id] = job_info
            
            start_time = time.time()
            try:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                for line in process.stdout:
                    self.logger.info(f"Job {job_id} output: {line.strip()}")
                
                process.wait()
                exit_code = process.returncode
                
                if exit_code == 0:
                    job_info.status = JobStatus.COMPLETED
                else:
                    job_info.status = JobStatus.FAILED
                    self.logger.error(f"Job {job_id} failed with exit code {exit_code}")
            except Exception as e:
                self.logger.error(f"Job {job_id} failed: {str(e)}")
                job_info.status = JobStatus.FAILED
                exit_code = 1
            finally:
                end_time = time.time()
                job_info.metrics = JobMetrics(
                    start_time=start_time,
                    end_time=end_time,
                    cpu_usage=0,  # Placeholder, implement actual CPU usage tracking
                    memory_usage=0,  # Placeholder, implement actual memory usage tracking
                    exit_code=exit_code
                )

        ThreadPoolExecutor().submit(run_job)
        return job_id

    def check_job_status(self, job_id: str) -> JobStatus:
        job_info = self.jobs.get(job_id)
        return job_info.status if job_info else JobStatus.FAILED

    def cancel_job(self, job_id: str) -> bool:
        job_info = self.jobs.get(job_id)
        if job_info and job_info.status == JobStatus.RUNNING:
            # Implement job cancellation logic here
            job_info.status = JobStatus.CANCELLED
            return True
        return False

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        job_info = self.jobs.get(job_id)
        if job_info and job_info.status == JobStatus.COMPLETED:
            return {
                "job_id": job_info.id,
                "status": job_info.status.value,
                "metrics": asdict(job_info.metrics) if job_info.metrics else None
            }
        return {}

    def get_job_metrics(self, job_id: str) -> JobMetrics:
        job_info = self.jobs.get(job_id)
        if job_info and job_info.metrics:
            return job_info.metrics
        raise ValueError(f"No metrics available for job {job_id}")


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