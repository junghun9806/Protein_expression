Directory Name Change & Description: 
Truncation --> FB
Finite State Projection (FSP) which simply block transition rate to the abandoned states. 
This model is called Finite Buffer model (FB).

FSP_Diag 
Finite State Projection (FSP) considers all the abandoned states into one sigle states. (denoted as g)
Here Diagonalization of the transition matrix is used to solve the CME (Chemical Master Equation). 
There should be trivial solution of steady states: 
P(g) == 1
P = 0 for all the other states

FSP_Diag_M1 
Finite State Projection Modified no. 1(FSP_Diag_M1) considers all the abandoned states into one sigle states. (denoted as g) 
However in this model, there is transition from state g to the boundary states. This prevents from the model to have trivial steady state solution. 
Yet, it is hard to justify the modification of the model.  Numerical simulation to show the performance is needed. 

Things to Do 
stationary Finite State Projection (sFSP)

FSP

Gillespie 

Linear Noise Approximation (LNA)