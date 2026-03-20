"""
Genetic Algorithm-Optimized Fuzzy Inference System for SOC Estimation.

Combines fuzzy logic (interpretable rule-based reasoning) with
genetic algorithm optimization of membership function parameters.

Aligned with IISER Bhopal internship work on genetic-fuzzy methods
for Li-ion battery SOC estimation.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Membership functions
# ---------------------------------------------------------------------------


def triangular_mf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership function: parameters (a, b, c) with a <= b <= c."""
    return np.maximum(
        0, np.minimum((x - a) / max(b - a, 1e-10), (c - x) / max(c - b, 1e-10))
    )


def gaussian_mf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Gaussian membership function."""
    return np.exp(-0.5 * ((x - mean) / max(sigma, 1e-10)) ** 2)


def trapezoidal_mf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoidal membership function: (a, b, c, d) with a <= b <= c <= d."""
    return np.maximum(
        0,
        np.minimum(
            np.minimum((x - a) / max(b - a, 1e-10), 1),
            (d - x) / max(d - c, 1e-10),
        ),
    )


# ---------------------------------------------------------------------------
# Fuzzy inference system
# ---------------------------------------------------------------------------


@dataclass
class FuzzyVariable:
    """Fuzzy linguistic variable with membership functions."""

    name: str
    universe: np.ndarray  # Range of the variable
    terms: dict = field(default_factory=dict)  # term_name -> mf_params

    def add_term(self, name: str, mf_type: str, params: list):
        self.terms[name] = {"type": mf_type, "params": params}

    def fuzzify(self, x: float) -> dict:
        """Compute membership degrees for all terms."""
        memberships = {}
        x_arr = np.array([x])
        for term_name, mf_info in self.terms.items():
            if mf_info["type"] == "triangular":
                memberships[term_name] = triangular_mf(x_arr, *mf_info["params"])[0]
            elif mf_info["type"] == "gaussian":
                memberships[term_name] = gaussian_mf(x_arr, *mf_info["params"])[0]
            elif mf_info["type"] == "trapezoidal":
                memberships[term_name] = trapezoidal_mf(x_arr, *mf_info["params"])[0]
        return memberships


@dataclass
class FuzzyRule:
    """IF-THEN fuzzy rule."""

    antecedents: dict  # {variable_name: term_name}
    consequent_term: str
    weight: float = 1.0


class MamdaniFIS:
    """Mamdani-type fuzzy inference system for SOC estimation."""

    def __init__(self):
        self.input_vars: dict[str, FuzzyVariable] = {}
        self.output_var: Optional[FuzzyVariable] = None
        self.rules: list[FuzzyRule] = []

    def add_input(self, var: FuzzyVariable):
        self.input_vars[var.name] = var

    def set_output(self, var: FuzzyVariable):
        self.output_var = var

    def add_rule(self, rule: FuzzyRule):
        self.rules.append(rule)

    def infer(self, inputs: dict) -> float:
        """
        Run fuzzy inference for given crisp inputs.

        Steps:
        1. Fuzzify inputs
        2. Evaluate rules (AND = min)
        3. Aggregate outputs
        4. Defuzzify (centroid method)
        """
        # 1. Fuzzify
        fuzzified = {}
        for var_name, value in inputs.items():
            if var_name in self.input_vars:
                fuzzified[var_name] = self.input_vars[var_name].fuzzify(value)

        # 2. Evaluate rules
        rule_strengths = {}  # consequent_term -> max firing strength
        for rule in self.rules:
            # AND: minimum of antecedent membership degrees
            strength = rule.weight
            for var_name, term_name in rule.antecedents.items():
                if var_name in fuzzified and term_name in fuzzified[var_name]:
                    strength = min(strength, fuzzified[var_name][term_name])
                else:
                    strength = 0
                    break

            term = rule.consequent_term
            if term not in rule_strengths:
                rule_strengths[term] = 0
            rule_strengths[term] = max(rule_strengths[term], strength)

        # 3 & 4. Defuzzify (centroid)
        if self.output_var is None:
            return 0.0

        universe = self.output_var.universe
        aggregated = np.zeros_like(universe, dtype=float)

        for term_name, strength in rule_strengths.items():
            if term_name in self.output_var.terms:
                mf_info = self.output_var.terms[term_name]
                if mf_info["type"] == "triangular":
                    mf_values = triangular_mf(universe, *mf_info["params"])
                elif mf_info["type"] == "gaussian":
                    mf_values = gaussian_mf(universe, *mf_info["params"])
                else:
                    mf_values = triangular_mf(universe, *mf_info["params"])

                # Clip by rule strength
                clipped = np.minimum(mf_values, strength)
                aggregated = np.maximum(aggregated, clipped)

        # Centroid defuzzification
        total = np.sum(aggregated)
        if total == 0:
            return np.mean(universe)
        centroid = np.sum(universe * aggregated) / total
        return float(centroid)


# ---------------------------------------------------------------------------
# Default FIS for SOC estimation
# ---------------------------------------------------------------------------


def create_default_soc_fis(mf_params: Optional[dict] = None) -> MamdaniFIS:
    """
    Create a default fuzzy system for SOC estimation from voltage and current.

    Input variables: voltage (V), current (A)
    Output: SOC (0-1)
    """
    fis = MamdaniFIS()

    params = mf_params or {}

    # Input: Voltage (2.5 - 4.2 V)
    voltage = FuzzyVariable("voltage", np.linspace(2.5, 4.2, 200))
    voltage.add_term("low", "triangular", params.get("v_low", [2.5, 2.5, 3.2]))
    voltage.add_term("medium", "triangular", params.get("v_medium", [3.0, 3.5, 4.0]))
    voltage.add_term("high", "triangular", params.get("v_high", [3.6, 4.2, 4.2]))
    fis.add_input(voltage)

    # Input: Current (0 - 4 A discharge)
    current = FuzzyVariable("current", np.linspace(0, 4, 200))
    current.add_term("low", "triangular", params.get("i_low", [0, 0, 1.5]))
    current.add_term("medium", "triangular", params.get("i_medium", [0.5, 2.0, 3.5]))
    current.add_term("high", "triangular", params.get("i_high", [2.5, 4.0, 4.0]))
    fis.add_input(current)

    # Output: SOC (0 - 1)
    soc_out = FuzzyVariable("soc", np.linspace(0, 1, 200))
    soc_out.add_term("very_low", "triangular", params.get("s_vlow", [0.0, 0.0, 0.2]))
    soc_out.add_term("low", "triangular", params.get("s_low", [0.1, 0.25, 0.4]))
    soc_out.add_term("medium", "triangular", params.get("s_medium", [0.3, 0.5, 0.7]))
    soc_out.add_term("high", "triangular", params.get("s_high", [0.6, 0.75, 0.9]))
    soc_out.add_term("very_high", "triangular", params.get("s_vhigh", [0.8, 1.0, 1.0]))
    fis.set_output(soc_out)

    # Rules (domain knowledge)
    rules = [
        FuzzyRule({"voltage": "high", "current": "low"}, "very_high"),
        FuzzyRule({"voltage": "high", "current": "medium"}, "high"),
        FuzzyRule({"voltage": "high", "current": "high"}, "medium"),
        FuzzyRule({"voltage": "medium", "current": "low"}, "high"),
        FuzzyRule({"voltage": "medium", "current": "medium"}, "medium"),
        FuzzyRule({"voltage": "medium", "current": "high"}, "low"),
        FuzzyRule({"voltage": "low", "current": "low"}, "low"),
        FuzzyRule({"voltage": "low", "current": "medium"}, "very_low"),
        FuzzyRule({"voltage": "low", "current": "high"}, "very_low"),
    ]
    for rule in rules:
        fis.add_rule(rule)

    return fis


# ---------------------------------------------------------------------------
# Genetic Algorithm Optimization
# ---------------------------------------------------------------------------


def _encode_fis_params(fis: MamdaniFIS) -> np.ndarray:
    """Encode FIS membership function parameters as a flat chromosome."""
    genes = []
    for var in list(fis.input_vars.values()) + [fis.output_var]:
        for term_name, mf_info in var.terms.items():
            genes.extend(mf_info["params"])
    return np.array(genes)


def _decode_fis_params(chromosome: np.ndarray, template_fis: MamdaniFIS) -> dict:
    """Decode chromosome back to FIS parameter dict."""
    params = {}
    idx = 0
    key_map = {
        "voltage": {"low": "v_low", "medium": "v_medium", "high": "v_high"},
        "current": {"low": "i_low", "medium": "i_medium", "high": "i_high"},
        "soc": {
            "very_low": "s_vlow",
            "low": "s_low",
            "medium": "s_medium",
            "high": "s_high",
            "very_high": "s_vhigh",
        },
    }

    for var in list(template_fis.input_vars.values()) + [template_fis.output_var]:
        for term_name, mf_info in var.terms.items():
            n_params = len(mf_info["params"])
            param_values = chromosome[idx : idx + n_params].tolist()
            # Ensure ordering for triangular MFs: a <= b <= c
            param_values.sort()
            key = key_map.get(var.name, {}).get(term_name, f"{var.name}_{term_name}")
            params[key] = param_values
            idx += n_params

    return params


def optimize_fis_genetic(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    voltage_idx: int = 0,
    current_idx: int = 1,
    pop_size: int = 50,
    n_generations: int = 30,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    seed: int = 42,
) -> dict:
    """
    Optimize FIS membership function parameters using a genetic algorithm.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True SOC values.
    feature_names : list
        Column names (to find voltage, current indices).
    pop_size : int
        Population size.
    n_generations : int
        Number of generations.

    Returns
    -------
    dict
        Best FIS, parameters, and optimization history.
    """
    rng = np.random.default_rng(seed)
    logger.info("Optimizing FIS with GA (pop=%d, gen=%d)...", pop_size, n_generations)

    # Determine column indices
    if "voltage" in feature_names:
        voltage_idx = feature_names.index("voltage")
    if "current" in feature_names:
        current_idx = feature_names.index("current")

    # Template FIS and initial chromosome
    template_fis = create_default_soc_fis()
    template_chromosome = _encode_fis_params(template_fis)
    n_genes = len(template_chromosome)

    # Initialize population
    population = []
    for _ in range(pop_size):
        # Perturb template
        noise = rng.uniform(-0.2, 0.2, n_genes)
        individual = np.clip(template_chromosome + noise, 0, 5)
        population.append(individual)

    # Fitness function (negative RMSE)
    def evaluate_fitness(chromosome):
        params = _decode_fis_params(chromosome, template_fis)
        fis = create_default_soc_fis(params)

        # Subsample for speed
        n_eval = min(len(X), 500)
        indices = rng.choice(len(X), n_eval, replace=False)

        predictions = []
        for idx in indices:
            pred = fis.infer(
                {
                    "voltage": float(X[idx, voltage_idx]),
                    "current": float(X[idx, current_idx]),
                }
            )
            predictions.append(pred)

        predictions = np.array(predictions)
        actual = y[indices]
        rmse = np.sqrt(np.mean((actual - predictions) ** 2))
        return -rmse  # Negative because we maximize fitness

    # Evolution loop
    history = []
    best_fitness = -np.inf
    best_chromosome = template_chromosome.copy()

    for gen in range(n_generations):
        # Evaluate fitness
        fitness_scores = np.array([evaluate_fitness(ind) for ind in population])

        # Track best
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_chromosome = population[gen_best_idx].copy()

        history.append(
            {
                "generation": gen + 1,
                "best_fitness": -best_fitness,  # Convert back to RMSE
                "avg_fitness": -np.mean(fitness_scores),
                "std_fitness": np.std(fitness_scores),
            }
        )

        if (gen + 1) % 5 == 0:
            logger.info(
                "Gen %d/%d: Best RMSE=%.4f, Avg RMSE=%.4f",
                gen + 1,
                n_generations,
                -best_fitness,
                -np.mean(fitness_scores),
            )

        # Selection (tournament)
        new_population = [best_chromosome.copy()]  # Elitism
        while len(new_population) < pop_size:
            # Tournament selection
            idx1, idx2 = rng.choice(pop_size, 2, replace=False)
            parent1 = (
                population[idx1]
                if fitness_scores[idx1] > fitness_scores[idx2]
                else population[idx2]
            )

            idx3, idx4 = rng.choice(pop_size, 2, replace=False)
            parent2 = (
                population[idx3]
                if fitness_scores[idx3] > fitness_scores[idx4]
                else population[idx4]
            )

            # Crossover
            if rng.random() < crossover_rate:
                cx_point = rng.integers(1, n_genes)
                child = np.concatenate([parent1[:cx_point], parent2[cx_point:]])
            else:
                child = parent1.copy()

            # Mutation
            for g in range(n_genes):
                if rng.random() < mutation_rate:
                    child[g] += rng.normal(0, 0.1)
                    child[g] = max(0, child[g])

            new_population.append(child)

        population = new_population[:pop_size]

    # Build best FIS
    best_params = _decode_fis_params(best_chromosome, template_fis)
    best_fis = create_default_soc_fis(best_params)

    logger.info("GA complete: Best RMSE=%.4f", -best_fitness)

    return {
        "fis": best_fis,
        "best_params": best_params,
        "best_rmse": round(-best_fitness, 4),
        "history": pd.DataFrame(history),
        "best_chromosome": best_chromosome,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_soc_fuzzy(
    fis: MamdaniFIS,
    X: np.ndarray,
    voltage_idx: int = 0,
    current_idx: int = 1,
) -> np.ndarray:
    """Predict SOC for all samples using the fuzzy system."""
    predictions = []
    for i in range(len(X)):
        pred = fis.infer(
            {
                "voltage": float(X[i, voltage_idx]),
                "current": float(X[i, current_idx]),
            }
        )
        predictions.append(pred)
    return np.array(predictions)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genetic-Fuzzy SOC Estimation")
    parser.add_argument(
        "--input", type=str, default="data/processed/battery_features.csv"
    )
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    from .feature_engineering import get_feature_columns

    df = pd.read_csv(args.input)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["soc"].values

    result = optimize_fis_genetic(
        X,
        y,
        feature_cols,
        pop_size=args.pop_size,
        n_generations=args.generations,
    )

    print(f"\nBest RMSE: {result['best_rmse']:.4f}")
    print("\nOptimization History:")
    print(result["history"].tail(10).to_string(index=False))

    from pathlib import Path

    Path(args.output).mkdir(parents=True, exist_ok=True)
    result["history"].to_csv(f"{args.output}/ga_history.csv", index=False)
