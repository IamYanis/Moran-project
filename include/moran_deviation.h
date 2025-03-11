#ifndef MORAN_DEVIATION_H
#define MORAN_DEVIATION_H

// Function prototype for solving the ODE
void solve_ode(double *ode_t, double *ode_x, int steps, double r, double x0);

// Function prototype for computing deviation
void compute_moran_deviation(int nsim, int *N_values, int num_N, double *r_values, int num_r, const char *filename);

#endif
