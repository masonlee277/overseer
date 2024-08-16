import json
import subprocess
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import os
import sys
import io

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import psutil
import threading 


from overseer.utils.logging import OverseerLogger
from overseer.config.config import OverseerConfig
from overseer.core.models import JobStatus, SimulationResult




class LocalEnvironment:
    def __init__(self, logger, log_simulation_output: bool = False):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.log_simulation_output = log_simulation_output

        if logger:
            self.logger = logger
        else:
            self.logger = OverseerLogger.getLogger(__name__)
            self.logger.setLevel(OverseerLogger.DEBUG)
            handler = OverseerLogger.StreamHandler()
            formatter = OverseerLogger.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
    def submit_job(self, command: str) -> str:
        job_id = str(uuid.uuid4())
        self.logger.info(f"Submitting job {job_id} with command: {command}")
        
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.jobs[job_id] = {
                "process": process,
                "status": JobStatus.RUNNING,
                "start_time": time.time(),
                "end_time": None,
                "cpu_usage": [],
                "memory_usage": [],
                "output": io.StringIO()
            }
            
            self.logger.debug(f"Job {job_id} started with PID {process.pid}")
            
            # Start monitoring the process
            threading.Thread(target=self._monitor_job, args=(job_id,), daemon=True).start()
            # Start capturing the output
            threading.Thread(target=self._capture_output, args=(job_id,), daemon=True).start()
        except Exception as e:
            self.logger.error(f"Error submitting job {job_id}: {str(e)}")
            self.jobs[job_id] = {
                "process": None,
                "status": JobStatus.FAILED,
                "start_time": time.time(),
                "end_time": time.time(),
                "cpu_usage": [],
                "memory_usage": [],
                "output": io.StringIO(f"Failed to start: {str(e)}")
            }
        
        return job_id


    def _capture_output(self, job_id: str) -> None:
        job = self.jobs[job_id]
        process = job["process"]
        output_buffer = job["output"]

        for line in iter(process.stdout.readline, ''):
            if self.log_simulation_output:
                self.logger.info(f"Job {job_id} output: {line.strip()}")
            output_buffer.write(line)

        process.stdout.close()

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        self.logger.debug(f"Retrieving results for job {job_id}")
        job = self.jobs[job_id]
        duration = round((job["end_time"] - job["start_time"]), 1) if job["end_time"] else None
        return {
            "status": job["status"].value,
            "duration": duration,
            "cpu_usage": sum(job["cpu_usage"]) / len(job["cpu_usage"]) if job["cpu_usage"] else 0,
            "memory_usage": sum(job["memory_usage"]) / len(job["memory_usage"]) if job["memory_usage"] else 0,
            "exit_code": job["process"].returncode if job["process"] else None
        }


    def _monitor_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        process = job["process"]
        
        while process.poll() is None:
            try:
                proc = psutil.Process(process.pid)
                cpu_percent = proc.cpu_percent(interval=1)
                memory_percent = proc.memory_percent()
                
                job["cpu_usage"].append(cpu_percent)
                job["memory_usage"].append(memory_percent)
                
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

    def check_job_status(self, job_id: str) -> JobStatus:
        self.logger.debug(f"Checking status for job {job_id}")
        return self.jobs[job_id]["status"]

    def cancel_job(self, job_id: str) -> bool:
        self.logger.info(f"Attempting to cancel job {job_id}")
        job = self.jobs[job_id]
        if job["process"] and job["status"] == JobStatus.RUNNING:
            job["process"].terminate()
            job["status"] = JobStatus.CANCELLED
            self.logger.info(f"Job {job_id} cancelled")
            return True
        return False



class ComputeManager:
    def __init__(self, config: OverseerConfig):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.log_simulation_output = self.config.get('log_simulation_output', False)

        self.environment = self._get_compute_environment()
        self.sim_dir = Path(self.config.get('directories', {}).get('elmfire_sim_dir', ''))
        self.start_script = self.sim_dir / self.config.get('directories', {}).get('start_script', '01-run.sh')
        
        if not self.start_script.exists():
            raise FileNotFoundError(f"Start script not found: {self.start_script}")
        
        self.logger.info(f"Initialized ComputeManager with simulation directory: {self.sim_dir}")
        self.logger.info(f"Using start script: {self.start_script}")

    def _get_compute_environment(self):
        env_type = self.config.get('compute_environment', 'local')
        if env_type == 'local':
            return LocalEnvironment(self.logger, self.log_simulation_output)
        else:
            raise ValueError(f"Unsupported compute environment: {env_type}")


    def submit_simulation(self) -> SimulationResult:
        command = f"cd {self.sim_dir} && bash {self.start_script.name}"
        self.logger.info(f"Preparing to submit simulation with command: {command}")
        
        try:
            job_id = self.environment.submit_job(command)
            self.logger.info(f"Submitted simulation job {job_id}")
            
            # Wait for the simulation to complete
            final_result = self.wait_for_simulation(job_id)
            return final_result
        except Exception as e:
            self.logger.error(f"Error submitting simulation: {str(e)}")
            return SimulationResult(job_id="", status=JobStatus.FAILED, error_message=str(e))

    def wait_for_simulation(self, job_id: str, check_interval: int = 2) -> SimulationResult:
        last_print_time = time.time()
        while True:
            status = self.check_simulation_status(job_id)
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                self.logger.info(f"Simulation {job_id} finished with status: {status.value}")
                results = self.retrieve_simulation_results(job_id)
                return results
            
            current_time = time.time()
            if current_time - last_print_time >= 8:
                self.logger.info(f"Simulation {job_id} is still running...")
                last_print_time = current_time
            
            time.sleep(check_interval)
            
    def retrieve_simulation_results(self, job_id: str) -> SimulationResult:
        results = self.environment.retrieve_results(job_id)
        self.logger.info(f"Retrieved results for job {job_id}: {json.dumps(results, indent=2)}")
        return SimulationResult(
            job_id=job_id,
            status=JobStatus(results['status']),
            duration=results.get('duration'),
            cpu_usage=results.get('cpu_usage'),
            memory_usage=results.get('memory_usage'),
            exit_code=results.get('exit_code')
        )


    def check_simulation_status(self, job_id: str) -> JobStatus:
        return self.environment.check_job_status(job_id)

    def cancel_simulation(self, job_id: str) -> bool:
        return self.environment.cancel_job(job_id)


    def set_log_simulation_output(self, value: bool):
        self.log_simulation_output = value
        if isinstance(self.environment, LocalEnvironment):
            self.environment.log_simulation_output = value
        self.logger.info(f"Simulation output logging set to: {value}")
def main():
    logger = OverseerLogger().get_logger('ElmfireComputeManagerTest')
    
    try:
        # Initialize the configuration
        config_path = Path(__file__).parent.parent / 'config' / 'elmfire_config.yaml'
        config = OverseerConfig(config_path)
        logger.info(f"Configuration loaded from {config_path}")

        # Initialize the ComputeManager
        try:
            compute_manager = ComputeManager(config)
            logger.info("ComputeManager initialized successfully")
        except FileNotFoundError as e:
            logger.error(f"Error initializing ComputeManager: {e}")
            sys.exit(1)

        # Submit a simulation
        submit_result = compute_manager.submit_simulation()
        if submit_result.status == JobStatus.FAILED:
            logger.error(f"Failed to submit simulation: {submit_result.error_message}")
            sys.exit(1)
        logger.info(f"Submitted simulation job with ID: {submit_result.job_id}")

        # Wait for the simulation to complete
        final_result = compute_manager.wait_for_simulation(submit_result.job_id)
        logger.info(f"Simulation completed with status: {final_result.status}")

        # Log the results
        if final_result.status == JobStatus.COMPLETED:
            logger.info("Simulation completed successfully")
            logger.info(f"Duration: {final_result.duration} seconds")
            logger.info(f"CPU Usage: {final_result.cpu_usage}%")
            logger.info(f"Memory Usage: {final_result.memory_usage}%")
            logger.info(f"Exit Code: {final_result.exit_code}")
        else:
            logger.warning(f"Simulation did not complete successfully. Status: {final_result.status}")
            if final_result.error_message:
                logger.error(f"Error message: {final_result.error_message}")

    except Exception as e:
        logger.critical(f"Unexpected error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()