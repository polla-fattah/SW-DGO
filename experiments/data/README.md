# Master Genuine Simulation Benchmark & Statistical Report

## 1. Benchmark Comparison (N=100 Genuine Trials per Algorithm)

| Algorithm | Success Rate | Makespan (s) [Mean ± SD] | [95% CI] | Personal Space Violations | Welch's p-value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D2RO (SW-DGO Proposed)** | 97.0% | 21.47 ± 5.32 | [20.42, 22.53] | 0.59 ± 5.90 | Baseline (N/A) |
| **Static A*** | 100.0% | 0.80 ± 0.00 | [0.80, 0.80] | 4.00 ± 0.00 | p < 0.001 |
| **Reactive Avoidance (Potential Field)** | 0.0% | 35.00 ± 0.00 | [35.00, 35.00] | 226.39 ± 69.25 | p < 0.001 |
| **Reactive ORCA (Velocity Obstacles)** | 0.0% | 35.00 ± 0.00 | [35.00, 35.00] | 21.51 ± 70.93 | p < 0.001 |
| **Decentralized Local MAPF** | 0.0% | 35.00 ± 0.00 | [35.00, 35.00] | 102.44 ± 15.00 | p < 0.001 |

## 2. Component Ablation Analysis (N=100 Genuine Trials per Configuration)

| Configuration | Omitted Component | Success Rate | Makespan (s) | Discomfort Integral | Shelf Scrapes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full D2RO Framework** | - | 99.0% | 22.98 ± 2.21 | 0.63 ± 3.09 | 0.00 ± 0.00 |
| **w/o V2V Mesh Telemetry** | - | 100.0% | 15.24 ± 1.95 | 0.43 ± 1.99 | 0.00 ± 0.00 |
| **w/o Corridor Mutex Lock** | - | 100.0% | 15.24 ± 1.95 | 0.43 ± 1.99 | 0.00 ± 0.00 |
| **w/o Human Gaussian Proxemics** | - | 17.0% | 32.70 ± 5.67 | 71.56 ± 34.26 | 0.00 ± 0.00 |
| **w/o Trolley Kinetic Safety Bubble** | - | 100.0% | 21.26 ± 2.17 | 0.41 ± 2.07 | 77.72 ± 11.94 |

## 3. Cross-Domain Generalization (N=100 Genuine Trials per Domain)

| Environment Domain | Success Rate | Makespan (s) | Mean Transit Time (s) | V2V Packets | Replans |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Retail Supermarket** | 100.0% | 23.07 ± 2.47 | 12.21 ± 0.62 | 15.2 ± 2.6 | 341.7 ± 82.8 |
| **Clinical Hospital** | 92.0% | 27.68 ± 11.35 | 19.60 ± 7.07 | 2.8 ± 2.0 | 309.6 ± 127.1 |
| **Airport Terminal** | 80.0% | 25.29 ± 13.60 | 14.32 ± 4.34 | 6.6 ± 14.8 | 897.9 ± 342.9 |
