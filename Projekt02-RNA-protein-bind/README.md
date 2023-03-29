# RNA-protein binding prediction

## Initialize environment
```bash
pip install -r requirements.txt
```

## Running analysis
Command:

```bash
python run_analysis.py [--clusters_path <path_to_clusters>]
```
* `--clusters_path`: Path to `.json` file with saved clustering results. If not provided user is guided through interactive clustering method tuning and selection.

To reproduce saved analysis use:
```bash
python run_analysis.py --clusters_path clusters/Clusters_2023_03_29_1713.json
```

