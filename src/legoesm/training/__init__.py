"""Training infrastructure for differentiable dycore + WeatherBench.

Provides three training modes:
1. Physics parameter tuning via gradient descent through the dycore
2. Neural GCM: neural network coupled to the dycore for physics
3. SFNO coupled to dycore: spherical Fourier neural operator + dynamics
"""
