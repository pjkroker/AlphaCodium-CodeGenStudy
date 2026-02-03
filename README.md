# Study LLM Code Generation
Fork of Alpha Codium. See original [github](https://github.com/Codium-ai/AlphaCodium) or [README](./README-alpha-codium.md).

## Installation
```
conda create --name alpha_codium-study python=3.10
conda activate alpha_codium-study
pip install -e .
```
## Usage
### solve dataset 
```python
from alpha_codium.gen.dataset_solver import solve_dataset
from alpha_codium.log import setup_logger
setup_logger()
from pathlib import Path

path_output = Path("results.jsonl")
solve_dataset(
    dataset_name=dataset_name,
    split_name=split_name,
    database_solution_path=database_solution_path,
    path_output=path_output
)
```

### solve problem
```python 
from alpha_codium.gen.coding_competitor import solve_problem
from alpha_codium.log import setup_logger
setup_logger()  # same as CLI does
from pathlib import Path

path_output = Path("results.jsonl")
solve_problem(
        dataset_name=dataset_name,
        split_name=split_name,
        problem_number=problem_number,
        problem_name=problem_name,
        path_output=path_output,
    )
```