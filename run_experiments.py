import os
from benchmarks.systems.pendulum import run_statistical_benchmark as run_pendulum
from benchmarks.systems.arm import run_statistical_benchmark as run_arm
import traceback

def main():
    print("Starting ESN Controller Experiments")
    os.makedirs("docs/figures", exist_ok=True)
    os.makedirs("benchmarks/results", exist_ok=True)
    os.makedirs("trained_models", exist_ok=True)

    try:
        print("\nRunning Pendulum Benchmark...")
        run_pendulum()
    except Exception as e:
        print(f"Error in Pendulum Benchmark: {e}")
        traceback.print_exc()

    try:
        print("\nRunning Robotic Arm Benchmark...")
        run_arm()
    except Exception as e:
        print(f"Error in Robotic Arm Benchmark: {e}")
        traceback.print_exc()

    print("\nExperiments completed. Results saved in benchmarks/results/ and plots in docs/figures/.")

if __name__ == "__main__":
    main()