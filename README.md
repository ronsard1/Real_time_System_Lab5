# Real-Time Systems – Lab 5 (Unit 5 Mini Projects)

University of Rwanda, College of Science and Technology, School of ICT, Department of Computer Engineering
Group 3: Ronsard RUGIRANGOGA (223026673), Ivan MULISA (223026695), Dieudonne BYIRINGIRO (223026513)

The written report summarises the results; this repository holds the code, the printed results, the CSV exports and the charts.

## Structure

```
Mini_Project_5_1_TMR_Watchdog/
    reliability_sim.py       hour-step simplex and TMR with transient / permanent faults, empirical vs analytical
                             MTBF and availability, watchdog MTTR, timeout sweep, lab tasks 1-4 -> results_5_1.txt, figures/
Mini_Project_5_2_Monte_Carlo_NMR/
    monte_carlo_nmr.py       failure-rate sweep 1e-4 ... 1e-2 per hour for simplex, TMR, 5MR, 7MR (K-of-N voter),
                             adaptive trial counts with 95 % confidence intervals -> results_5_2.txt, sweep_results.csv, figures/
Mini_Project_5_3_Checkpointing/
    checkpoint_sim.py        checkpoint / rollback simulator, interval sweep for three MTBF values, Young and exact-model
                             optima -> results_5_3.txt, completion_times.csv, figures/
```

## Run

Python 3.8 or later; matplotlib is optional (only for the charts).

```
python Mini_Project_5_1_TMR_Watchdog/reliability_sim.py       (~20 s)
python Mini_Project_5_2_Monte_Carlo_NMR/monte_carlo_nmr.py    (~25 s)
python Mini_Project_5_3_Checkpointing/checkpoint_sim.py       (~10 s)
```

All seeds are fixed, so the output matches the report.
