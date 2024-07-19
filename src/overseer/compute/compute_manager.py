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

    def submit_job(self, command: str, resources: JobResources) -> str:
        job_id = str(uuid.uuid4())
        self.logger.info(f"Submitting local job {job_id}")
        
        def run_job():
            job_info = JobInfo(id=job_id, status=JobStatus.RUNNING, submit_time=time.time(), resources=resources)
            self.jobs[job_id] = job_info
            
            start_time = time.time()
            try:
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                job_info.status = JobStatus.COMPLETED
                exit_code = result.returncode
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Job {job_id} failed: {str(e)}")
                job_info.status = JobStatus.FAILED
                exit_code = e.returncode
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

class SlurmEnvironment(ComputeEnvironment):
    def __init__(self, logger: OverseerLogger):
        self.logger = logger

    def submit_job(self, command: str, resources: JobResources) -> str:
        self.logger.info("Submitting SLURM job")
        slurm_script = self._generate_slurm_script(command, resources)
        result = subprocess.run(['sbatch'], input=slurm_script, text=True, capture_output=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        self.logger.info(f"Submitted SLURM job {job_id}")
        return job_id

    def check_job_status(self, job_id: str) -> JobStatus:
        result = subprocess.run(['squeue', '-h', '-j', job_id, '-o', '%t'], capture_output=True, text=True)
        slurm_status = result.stdout.strip()
        if not slurm_status:
            return JobStatus.COMPLETED
        elif slurm_status == 'PD':
            return JobStatus.PENDING
        elif slurm_status == 'R':
            return JobStatus.RUNNING
        else:
            return JobStatus.FAILED

    def cancel_job(self, job_id: str) -> bool:
        try:
            subprocess.run(['scancel', job_id], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        metrics = self.get_job_metrics(job_id)
        return {
            "job_id": job_id,
            "status": self.check_job_status(job_id).value,
            "metrics": asdict(metrics)
        }

    def get_job_metrics(self, job_id: str) -> JobMetrics:
        result = subprocess.run(['sacct', '-j', job_id, '--format=Start,End,MaxRSS,ExitCode', '-n'], capture_output=True, text=True)
        metrics_raw = result.stdout.strip().split()
        return JobMetrics(
            start_time=time.mktime(time.strptime(metrics_raw[0], "%Y-%m-%dT%H:%M:%S")),
            end_time=time.mktime(time.strptime(metrics_raw[1], "%Y-%m-%dT%H:%M:%S")),
            cpu_usage=0,  # Placeholder, implement actual CPU usage retrieval
            memory_usage=float(metrics_raw[2][:-1]),  # Remove 'K' from MaxRSS
            exit_code=int(metrics_raw[3].split(':')[0])
        )

    def _generate_slurm_script(self, command: str, resources: JobResources) -> str:
        return f"""#!/bin/bash
#SBATCH --job-name=elmfire_sim
#SBATCH --output=elmfire_sim_%j.out
#SBATCH --error=elmfire_sim_%j.err
#SBATCH --time={resources.time_limit_hours}:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={resources.cpus}
#SBATCH --mem={resources.memory_gb}G
{'#SBATCH --gres=gpu:1' if resources.gpu else ''}

{command}
"""

class ComputeManager:
    def __init__(self, config: OverseerConfig):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.environment = self._get_compute_environment()

    def _get_compute_environment(self) -> ComputeEnvironment:
        env_type = self.config.get('compute_environment', 'local')
        if env_type == 'local':
            return LocalEnvironment(self.logger)
        elif env_type == 'slurm':
            return SlurmEnvironment(self.logger)
        else:
            raise ValueError(f"Unsupported compute environment: {env_type}")

    def submit_simulation(self, input_dir: Path, elmfire_version: str) -> str:
        command = f"elmfire_{elmfire_version} {input_dir}/elmfire.data"
        resources = JobResources(**self.config.get('compute_resources', {}))
        job_id = self.environment.submit_job(command, resources)
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
                return status
            time.sleep(check_interval)

    def add_compute_resources(self, resources: JobResources):
        # This method would be implemented differently for different environments
        # For now, we'll just log the request
        self.logger.info(f"Request to add compute resources: {asdict(resources)}")
        # In a real implementation, you might update a resource pool or
        # communicate with a cluster management system