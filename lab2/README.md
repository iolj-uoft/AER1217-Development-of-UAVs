# AER1217 Lab 2: Quadrotor Simulation and Control

## 1. Overview
This lab involves designing a **quadrotor position controller** and implementing it in a simulation using PyBullet. The quadrotor must track a **circular trajectory** using a **geometric controller**, which consists of:
- **Position control** (feedback + feed-forward terms)
- **Thrust computation**
- **Attitude control** (orientation computation)

## 2. Code Structure

### **Main Files**
- **`main.py`**: The entry point for running the simulation.
  - Loads environment and quadrotor parameters from `lab2.yaml`.
  - Initializes the planner and controller.
  - Executes the trajectory tracking loop.

- **`lab2.yaml`**: Configuration file for the quadrotor environment.
  - Defines the **circular trajectory parameters**.
  - Sets **simulation constraints** (tracking limits, control frequency, collision rules).
  - Provides **randomization settings** for robustness testing.

- **`planner.py`**: Generates the reference trajectory for the quadrotor.
  - Computes waypoints based on lab2.yaml.
  - **Modification:** Added `self.tolerance = initial_info.get("tracking_tolerance", 0.1)`, allowing users to specify tracking accuracy and prevent errors when initializing the `GeoController` class

### **Controller Components**
- **`edit_this.py`**: Implements the geometric position controller.
  - Contains `_compute_desired_force_and_euler()` for computing thrust and desired Euler angles.
  - **Modification:** Filled in missing computations for accurate force and orientation control.

- **`project_utils.py`**: Contains utility functions used throughout the simulation.
  - Provides helper functions for matrix operations, transformations, and UAV state processing.

## 3. Modifications & Improvements
### **1️. Completed the 3 tasks in `_compute_desired_force_and_euler()`**
- **File:** `edit_this.py`
- **Changes:**
  - Constructed a PD controller for thrust output.
  - Implemented $a_{des} = a_{fb} + a_{ref} + g\mathbf{z}_w$ (Task 1).
  - Completed thrust calculation using $c_{cmd} = m||\mathbf{a}_{des}||$(Equation 3). (Task 2)
  - Computed desired orientation using Equations (5)–(7) given in the LAB handout. (Task 3)
- **Impact:**
  - Enables proper UAV **position tracking and stability**.

### **2️. Added Tracking Tolerance in `planner.py`**
- **File:** `planner.py`
- **Modification:**
  - Added:
    ```python
    self.tolerance = initial_info.get("tracking_tolerance", 0.1)
    ```
  - Allows users to **customize the tracking tolerance**, and prevent `no attribute` error when initializing the `GeoController` class.
- **Impact:**
  - Showcase UAV trajectory tracking **adaptability**.