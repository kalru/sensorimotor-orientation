# Orientation Invariant Sensorimotor Object Recognition Using Cortical Grid Cells
## Setup
Run the following commands within your environment to install the required packages:

```
pip install -r .\requirements.txt

pip install -i https://test.pypi.org/simple/ htm.core==2.1.15
```
## Generating Figures
Use the [generate_figure.py](generate_figure.py) script to run the experiments and generate the figures presented in the paper:
```
usage: generate_figure.py [-h] [-c NUM] [-l] [FIGURE]
positional arguments:
  FIGURE                Specify the figure name to generate. Possible values
                        are: ['5A', '5B', '6', '7', '8']
optional arguments:
  -h, --help            show this help message and exit
  -c NUM, --cpuCount NUM
                        Limit number of cpu cores. Defaults to all available
                        cores
  -l, --list            List all figures
```
For example to run the convergence experiment presented in "*Figure 4A*" run the following command:

```
python generate_figure.py 4A
```
## How to Run Custom Experiments
The module 