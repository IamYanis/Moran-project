# Moran-project

1. (i) To study the supremum deviation: 
  -> Choose the size of the subdivision K of the set [0,T(N)] large enough

2. (i) To study histograms of the law of the fixation time:
  -> In main.c, choose K=1 to avoid useless computation and T_N big enough to cover the value studied of the law of the fixation time (eg. take T_N = 500 for r >= 1.1)

To create the data: 
  gcc -o moran_sim src/main.c -I"C:/msys64/mingw64/include" -L"C:/msys64/mingw64/lib" -lgsl -lgslcblas -lm

  ./moran_sim

1. (ii) To plot the deviation:
   -> in plot_devation.py, adapt n_sim to match N_SIM in main.c

   python scripts/plot_deviation.py

2. (ii) To plot the histograms of the fixation time:
   python scripts/plot_hist.py

3. To plot the trajectories and ODE:
  python scripts/plot_traj.py
