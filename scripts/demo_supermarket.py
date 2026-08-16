"""
Main Execution Script for SW-DGO Framework (D²RO).
100% Pure Python Simulation: Launches the Native Desktop GUI Simulator with interactive
scenario switcher (Scenarios A through E), live 60 FPS animation, and real-time telemetry.
"""

from d2ro.sim.gui import launch_gui

if __name__ == "__main__":
    print("Launching Native Python D²RO Fleet Simulator...")
    launch_gui()
