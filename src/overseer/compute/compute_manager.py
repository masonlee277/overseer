import uuid
import subprocess
import threading
import time
import psutil
from queue import Queue, Empty
from typing import Dict, Any, Optional
from enum import Enum
import json
import subprocess
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import os
import sys
import io
import traceback 

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import psutil
import threading 
import re

from overseer.utils.logging import OverseerLogger
from overseer.config.config import OverseerConfig
from overseer.core.models import JobStatus, SimulationResult
import threading
from queue import Queue, Empty


class SimulationOutput:
    def __init__(self):
        self.output_queue = Queue()
        self.is_complete = False
        self.full_output = []

    def add_line(self, line: str):
        """Add a line of output to the simulation output."""
        self.output_queue.put(line)
        self.full_output.append(line)

    def get_next_line(self, block: bool = False, timeout: Optional[float] = None) -> Optional[str]:
        """Get the next line of output from the simulation."""
        try:
            return self.output_queue.get(block=block, timeout=timeout)
        except Empty:
            return None

    def get_output(self, block: bool = False, timeout: Optional[float] = None) -> Optional[str]:
        """Get the next line of output from the simulation."""
        try:
            return self.output_queue.get(block=block, timeout=timeout)
        except Empty:
            return None
    def mark_as_complete(self):
        """Mark the simulation output as complete."""
        self.is_complete = True

    def get_all_output(self) -> str:
        """Get the full output of the simulation as a single string."""
        return "".join(self.full_output)

class LocalEnvironment:
    def __init__(self, logger):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.logger = logger

    def submit_job(self, command: str) -> str:
        """Submit a new job to the local environment."""
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
            
            simulation_output = SimulationOutput()
            
            self.jobs[job_id] = {
                "process": process,
                "status": JobStatus.RUNNING,
                "start_time": time.time(),
                "end_time": None,
                "cpu_usage": [],
                "memory_usage": [],
                "output": simulation_output
            }
            
            # Start output capture thread
            threading.Thread(target=self._capture_output, args=(job_id,), daemon=True).start()
            
            # Start resource monitoring thread
            threading.Thread(target=self._monitor_job, args=(job_id,), daemon=True).start()
            
            return job_id
        except Exception as e:
            self.logger.error(f"Error submitting job {job_id}: {str(e)}")
            simulation_output = SimulationOutput()
            simulation_output.add_line(f"Failed to start: {str(e)}")
            simulation_output.mark_as_complete()
            self.jobs[job_id] = {
                "process": None,
                "status": JobStatus.FAILED,
                "start_time": time.time(),
                "end_time": time.time(),
                "cpu_usage": [],
                "memory_usage": [],
                "output": simulation_output
            }
            return job_id

    def _capture_output(self, job_id: str) -> None:
        """Capture the output of a job and add it to the SimulationOutput."""
        job = self.jobs[job_id]
        process = job["process"]
        simulation_output = job["output"]

        for line in iter(process.stdout.readline, ''):
            self.logger.info(f"Job {job_id} output: {line.strip()}")
            simulation_output.add_line(line)

        process.stdout.close()
        simulation_output.mark_as_complete()

    def get_job_output(self, job_id: str) -> Optional[SimulationOutput]:
        """Get the SimulationOutput object for a specific job."""
        if job_id in self.jobs:
            return self.jobs[job_id]["output"]
        self.logger.error(f"Job {job_id} not found")
        return None

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve the final results of a job."""
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
        """Monitor the resource usage of a job."""
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
        """Finalize a job after it has completed or failed."""
        job = self.jobs[job_id]
        process = job["process"]
        
        job["end_time"] = time.time()
        job["status"] = JobStatus.COMPLETED if process.returncode == 0 else JobStatus.FAILED
        
        self.logger.info(f"Job {job_id} finished with status {job['status'].value}")
        self.logger.debug(f"Job {job_id} exit code: {process.returncode}")

    def check_job_status(self, job_id: str) -> JobStatus:
        """Check the current status of a job."""
        self.logger.debug(f"Checking status for job {job_id}")
        return self.jobs[job_id]["status"]

    def cancel_job(self, job_id: str) -> bool:
        """Attempt to cancel a running job."""
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
            self.logger.info("Using local environment")
            return LocalEnvironment(self.logger)
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
        simulation_output = self.environment.get_job_output(job_id)
        
        while True:
            status = self.check_simulation_status(job_id)
            
            # Process any new output
            while True:
                line = simulation_output.get_output(block=False)
                if line is None:
                    break
                error = self.detect_simulation_error(line)
                if error:
                    self.logger.error(f"Error detected in job {job_id}: {error}")
                    print(f"Error detected in job {job_id}: {error}.. Returning failed simulationResult.")  # Print the error
                    
                    return SimulationResult(
                        job_id=job_id,
                        status=JobStatus.FAILED,
                        error_message=error,
                        duration=None,
                        cpu_usage=None,
                        memory_usage=None,
                        exit_code=None
                    )

            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                self.logger.info(f"Simulation {job_id} finished with status: {status.value}")
                results = self.retrieve_simulation_results(job_id)
                
                # Final check for errors in the complete output
                if isinstance(self.environment, LocalEnvironment):
                    error = self.detect_simulation_error(simulation_output.get_full_output())
                    if error:
                        self.logger.error(f"Error detected in job {job_id}: {error}")
                        print(f"Error detected in job {job_id}: {error}")  # Print the error
                        return SimulationResult(
                            job_id=job_id,
                            status=JobStatus.FAILED,
                            error_message=error,
                            duration=results.duration,
                            cpu_usage=results.cpu_usage,
                            memory_usage=results.memory_usage,
                            exit_code=results.exit_code
                        )
                
                return results
            
            current_time = time.time()
            if current_time - last_print_time >= 8:
                self.logger.info(f"Simulation {job_id} is still running...")
                last_print_time = current_time
            
            time.sleep(check_interval)


    def detect_simulation_error(self, output: str) -> Optional[str]:
        self.logger.debug(f"Detecting simulation error in output: {output}")
        error_patterns = [
            r"ERROR",
            r"FAILURE",
            r"Exception",
            r"Traceback",
            r"No such file or directory",
            r"RuntimeError"
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                # Extract the line containing the error and some context
                lines = output.splitlines()
                error_line_index = next(i for i, line in enumerate(lines) if pattern.lower() in line.lower())
                context_start = max(0, error_line_index - 2)
                context_end = min(len(lines), error_line_index + 3)
                error_context = "\n".join(lines[context_start:context_end])
                return f"Error detected: {error_context}"
        
        return None        
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

    def build_elmfire(self) -> bool:
        print("Building ELMFIRE...")
        self.logger.info("Starting ELMFIRE build process")

        elmfire_base_dir = self.config.get('environment', {}).get('elmfire_base_dir')
        if not elmfire_base_dir:
            self.logger.error("ELMFIRE_BASE_DIR not set in configuration")
            return False

        build_command = f"cd {elmfire_base_dir}/build/linux && ./make_gnu.sh"
        self.logger.debug(f"Build command: {build_command}")

        try:
            process = subprocess.Popen(
                build_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Capture and log the output
            for line in iter(process.stdout.readline, ''):
                self.logger.info(f"ELMFIRE build: {line.strip()}")

            process.stdout.close()
            return_code = process.wait()

            if return_code == 0:
                self.logger.info("ELMFIRE build completed successfully")
                print("ELMFIRE build completed successfully")
                return True
            else:
                self.logger.error(f"ELMFIRE build failed with return code {return_code}")
                print(f"ELMFIRE build failed with return code {return_code}")
                return False

        except Exception as e:
            #log full traceback
            self.logger.error(f"Error during simulation run: {str(e)}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            print(f"Error during ELMFIRE build: {str(e)}")
            return False

        
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
            logger.error(f"Full traceback:")
            logger.error(traceback.format_exc())
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