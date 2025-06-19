import abc
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseExperiment(abc.ABC):
    """
    Abstract base class for experiments.

    This class defines the basic structure for experiments, including setup,
    run, and teardown phases. The execute method orchestrates these phases.
    """

    @abc.abstractmethod
    def setup(self):
        """
        Abstract method for setting up the experiment.

        This method should be implemented by subclasses to perform any necessary
        setup steps before the experiment runs.
        """
        pass

    @abc.abstractmethod
    def run(self):
        """
        Abstract method for running the experiment.

        This method should be implemented by subclasses to define the core logic
        of the experiment.
        """
        pass

    @abc.abstractmethod
    def teardown(self):
        """
        Abstract method for tearing down the experiment.

        This method should be implemented by subclasses to perform any necessary
        cleanup steps after the experiment has run.
        """
        pass

    def execute(self):
        """
        Executes the experiment by calling setup, run, and teardown in sequence.

        This method ensures that teardown is called even if setup or run
        raise an exception. It also logs the start and end of each phase.
        """
        self.results = {} # Initialize results dictionary
        try:
            logging.info("Setting up experiment...")
            self.setup()
            logging.info("Experiment setup complete.")

            logging.info("Running experiment...")
            self.run()
            logging.info("Experiment run complete.")
        except Exception as e:
            logging.error(f"Exception during experiment setup or run: {e}")
            self.results['error'] = str(e)
            self.results['error_type'] = type(e).__name__
            # self.results['elapsed_time'] = elapsed # This should already be there or handled by existing code
            raise
        finally:
            logging.info("Tearing down experiment...")
            self.teardown()
            logging.info("Experiment teardown complete.")

if __name__ == '__main__':
    # This is an example of how a subclass might be defined and used.
    # This part would typically be in a separate file.

    class MyExperiment(BaseExperiment):
        def setup(self):
            print("MyExperiment: Setup")
            # Simulate some setup work
            import time
            time.sleep(0.5)

        def run(self):
            print("MyExperiment: Run")
            # Simulate some work
            import time
            time.sleep(1)
            # Example of an error during run
            # raise ValueError("Something went wrong during run")


        def teardown(self):
            print("MyExperiment: Teardown")
            # Simulate some cleanup
            import time
            time.sleep(0.3)

    # Example usage:
    experiment = MyExperiment()
    experiment.execute()

    class FailingExperiment(BaseExperiment):
        def setup(self):
            print("FailingExperiment: Setup")
            raise RuntimeError("Setup failed!")

        def run(self):
            print("FailingExperiment: Run (should not be reached)")


        def teardown(self):
            print("FailingExperiment: Teardown (should still be called)")

    print("\n--- Running experiment that fails in setup ---")
    failing_experiment = FailingExperiment()
    failing_experiment.execute()

    class FailingRunExperiment(BaseExperiment):
        def setup(self):
            print("FailingRunExperiment: Setup")


        def run(self):
            print("FailingRunExperiment: Run")
            raise ValueError("Run failed!")


        def teardown(self):
            print("FailingRunExperiment: Teardown (should still be called)")

    print("\n--- Running experiment that fails in run ---")
    failing_run_experiment = FailingRunExperiment()
    failing_run_experiment.execute()
