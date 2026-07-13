# Algorithm Arena

Metaheuristic optimization algorithms (PSO, GA, SA, GWO, HHO, Cuckoo Search)
racing against each other on standard benchmark functions, with live convergence
visualization and user-defined objective functions.

## Status
🚧 Work in progress — Phase 5 (custom expression input) complete.

## Features
- 6 metaheuristic optimizers behind a shared `Optimizer` interface: PSO, Genetic
  Algorithm, Grey Wolf Optimizer, Harris Hawks Optimization, Simulated Annealing,
  Cuckoo Search
- 4 standard benchmark functions (Sphere, Rastrigin, Ackley, Rosenbrock)
- Live animated contour plots showing agent movement across iterations
- Convergence comparison charts across multiple algorithms
- Interactive Streamlit dashboard with Single Run and Race Mode
- **User-defined objective functions** — write your own expression (e.g.
  `x**2 + y**2 + 10*sin(x)`) and race optimizers on it

## 🚀 Live Demo
[Try it here](https://algorithm-arena-htquudbeabvpxlcvxyfzv5.streamlit.app/)

## Setup
...

## Setup
\`\`\`bash
uv sync
uv run pytest
\`\`\`

## Run the dashboard
\`\`\`bash
uv run streamlit run src/algorithm_arena/app/dashboard.py
# or
uv run algorithm-arena
\`\`\`

## Custom expressions

The dashboard lets you type your own 2D objective function instead of picking
a preset benchmark. Example: `x**2 + y**2 + 10*sin(x)`.

**Security note:** user input is never passed to Python's `eval()`. Expressions
are parsed with `sympy.parsing.sympy_parser.parse_expr`, restricted to:

- A character whitelist (regex) that rejects underscores and anything outside
  basic math syntax — this alone blocks dunder-based patterns like `__import__`.
- A `local_dict` limited to `x`, `y`, and a fixed set of math functions
  (`sin`, `cos`, `exp`, `sqrt`, ...) — any other name fails to resolve.
- `global_dict={"__builtins__": {}}` — removes access to real Python builtins
  during the internal `eval` that `sympy` performs, closing the main RCE vector.
- The `auto_number` transformation is disabled, since it conflicts with the
  builtins lockdown above; numeric literals are handled as plain Python
  numbers and wrapped into `sympy` types after parsing.

This was tested against injection attempts (e.g. `__import__('os').system(...)`)
during development — see `tests/test_custom_expression.py`.

## Project structure
\`\`\`
src/algorithm_arena/
├── optimizers/       # Optimizer interface + PSO, GA, GWO, HHO, SA, Cuckoo Search
├── benchmarks/        # Standard benchmark functions + safe custom expression parser
├── visualization/      # Plotly contour animations and convergence plots
└── app/               # Streamlit dashboard
\`\`\`

## Roadmap
- [x] Phase 0 — Project scaffolding
- [x] Phase 1 — Optimizer interface, benchmarks, PSO, GA
- [x] Phase 2 — GWO, HHO, Simulated Annealing, Cuckoo Search
- [x] Phase 3 — Contour animation + convergence comparison plots
- [x] Phase 4 — Streamlit dashboard (Single Run + Race Mode)
- [x] Phase 5 — User-defined objective functions (safe sympy parsing)
- [x] Phase 6 — Statistical comparison (multi-seed runs, boxplots, Wilcoxon test)
- [ ] Phase 7 — CI/CD, Docker, deployment
- [ ] Phase 8 — Final polish and presentation

## Statistical comparison methodology

Algorithms are compared using paired Wilcoxon signed-rank tests across N
independent runs (same seed sequence for every algorithm, ensuring paired
comparisons). Wilcoxon was chosen over a t-test because best-score
distributions from metaheuristics are typically non-normal (right-skewed),
which violates the t-test's core assumption — this is standard practice in
the metaheuristics literature.