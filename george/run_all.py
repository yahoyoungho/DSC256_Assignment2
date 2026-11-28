"""One-command reproduction from source data through the executed HTML report."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_*.py', '-v'], check=True)
    for script in ['experiments.py', 'diagnostics.py', 'verify.py', 'build_report.py']:
        subprocess.run([sys.executable, str(HERE / script)], check=True)
