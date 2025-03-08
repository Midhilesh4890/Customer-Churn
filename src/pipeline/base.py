"""
Base pipeline components for the churn prediction project.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineComponent(ABC):
    """Base class for all pipeline components."""

    def __init__(self, name: str):
        """
        Initialize a pipeline component.
        
        Args:
            name: Name of the component.
        """
        self.name = name
        self.logger = get_logger(f"{self.__class__.__name__}.{name}")

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Run the pipeline component.
        
        Returns:
            Any: The output of the component.
        """
        pass

    def log_execution_time(func):
        """Decorator to log the execution time of a component."""

        def wrapper(self, *args, **kwargs):
            self.logger.info(f"Starting {self.name} component")
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            self.logger.info(
                f"Completed {self.name} component in {execution_time:.2f} seconds")
            return result
        return wrapper


class Pipeline:
    """
    Main pipeline class to orchestrate the execution of multiple components.
    
    This class manages the flow between different pipeline components and
    handles the overall execution of the churn prediction pipeline.
    """

    def __init__(self, name: str, components: List[PipelineComponent] = None):
        """
        Initialize a pipeline.
        
        Args:
            name: Name of the pipeline.
            components: List of pipeline components to execute in order.
        """
        self.name = name
        self.components = components or []
        self.logger = get_logger(f"Pipeline.{name}")
        self.results = {}

    def add_component(self, component: PipelineComponent) -> 'Pipeline':
        """
        Add a component to the pipeline.
        
        Args:
            component: The component to add.
            
        Returns:
            Pipeline: The pipeline instance for chaining.
        """
        self.components.append(component)
        return self

    def run(self, input_data: Any = None) -> Dict[str, Any]:
        """
        Run the pipeline with all its components.
        
        Args:
            input_data: Initial input data for the pipeline.
            
        Returns:
            Dict[str, Any]: Dictionary of results from all components.
        """
        self.logger.info(f"Starting pipeline: {self.name}")
        start_time = time.time()

        # Initialize result with input data
        result = input_data

        # Execute each component in sequence
        for component in self.components:
            self.logger.info(f"Executing component: {component.name}")
            component_start_time = time.time()

            # Run the component with the result of the previous component
            result = component.run(result)

            # Store the result
            self.results[component.name] = result

            component_end_time = time.time()
            component_time = component_end_time - component_start_time
            self.logger.info(
                f"Component {component.name} completed in {component_time:.2f} seconds")

        end_time = time.time()
        total_time = end_time - start_time
        self.logger.info(
            f"Pipeline {self.name} completed in {total_time:.2f} seconds")

        return self.results

    def get_result(self, component_name: str) -> Any:
        """
        Get the result of a specific component.
        
        Args:
            component_name: Name of the component.
            
        Returns:
            Any: The result of the component.
            
        Raises:
            KeyError: If the component name is not found in the results.
        """
        if component_name not in self.results:
            raise KeyError(
                f"Component {component_name} not found in pipeline results")

        return self.results[component_name]
