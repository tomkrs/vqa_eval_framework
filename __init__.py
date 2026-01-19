"""Minimal Medical VQA evaluation framework.

Folders:
- datasets/: dataset adapters yielding standardized VQASample objects
- models/: model adapters exposing a common predict() interface
- process/: inference runner + evaluation utilities

Designed for easy use in a single Colab notebook.
"""
