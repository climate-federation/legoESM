"""WeatherBench2 evaluation framework for legoESM."""

from evaluations.metrics import rmse, acc, bias, spread_skill_ratio, compute_scorecard
from evaluations.weatherbench import WB2EvalConfig, evaluate_model
