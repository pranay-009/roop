#!/usr/bin/env python3

from roop import core
import argparse

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="A script that accepts source, target, and output paths.")

    # # Add arguments for source path, target path, and output path
    # parser.add_argument("source_path", type=str, help="Path to the source file or directory")
    # parser.add_argument("target_path", type=str, help="Path to the target file or directory")
    # parser.add_argument("output_path", type=str, help="Path to store the output file")

    # Parse the arguments
    # args = parser.parse_args()
    source_path = './data/girl.png'
    target_path = './data/indian.png'
    output_path = './data/swapped.png'
    core.run(source_path, target_path, output_path)
