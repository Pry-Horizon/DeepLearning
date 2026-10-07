"""Current server entry: official ResNet34 weights, then PyTorch baseline training."""
import subprocess
import sys
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    subprocess.run([sys.executable,str(root/'prepare_resnet34.py')],cwd=root,check=True)
    subprocess.run([sys.executable,str(root/'train.py'),*sys.argv[1:]],cwd=root,check=True)


if __name__=='__main__':main()
