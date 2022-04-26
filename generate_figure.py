import argparse

import algorithm_comparison
import orientation_selectivity
import multi_column_convergence
import capacity_test


RESULT_DIR_NAME = "results"
PLOTS_DIR_NAME = "plots"



def generateFigure5A(cpuCount):
  """
  The orientational selectivity of an equivalent module rotation of 144◦ is shown. 
  Approximately 73% of objects that had the highest firing rate in this direction 
  (estimated orientation) fell in the idealorientational receptive field of this 
  configuration. The normalised firing rate has a primary peak that corresponds 
  with the histogram.
  """
  orientation_selectivity.runExperimentA()



def generateFigure5B(cpuCount):
  """
  The orientational selectivity for a module rotation of approximately 259◦ is 
  shown, with 80% of object orientations correctly identified within the ideal 
  orientation range.
  """
  orientation_selectivity.runExperimentB()



def generateFigure6(cpuCount):
  """
  omparison between algorithms showing model accuracy for 25 grid cell modules 
  and 13 cells per axis. The experiment was repeated 10 times, with the 5th, 
  50th, and 95th percentiles shown. The base algorithm represents a network 
  with no orientation capabilities, while the ideal algorithm has prior 
  knowledge of the ideal equivalent orientation of its location layer for each 
  object. The accuracy has beentested for 1000 random rotations from a training 
  pool of 50 objects, and accumulates over 10 sensations as more objects are 
  recognised. The proposed algorithm (M = 0.948,SD = 0.014) has comparable
  accuracy to the ideal model for the given parameters, and improves 
  significantly from the baseline case
  (M = 0.780,SD = 0.0596), t(9) = 11.46,p < 0.0001.
  """
  algorithm_comparison.runExperiment()


def generateFigure7(cpuCount):
  """
  Comparison between different numbers of columns over a range of unique 
  feature pool sizes. More columns retain better accuracy as object ambiguity 
  increases (object uniqueness decreases). The network was configured to have 
  25 modules and 13 cells per axis, and was trained on 50 objects with 10 
  features each. The experiment was repeated 10 times, with the 5th, 50th, 
  and 95th percentiles shown.
  """
  multi_column_convergence.runExperiment()



def generateFigure8(cpuCount):
  """
  Comparison between the effects of cells per axis and number of modules on 
  model accuracy.Object ambiguity is adjusted by varying the amount of learned 
  objects for a fixed number of unique featuresof 40. Cells per axis is denoted 
  by colour and number of modules is shown by line shape. The illustration shows 
  that generally models with more cells per module have better accuracy for more 
  ambiguous objects, but only if they have enough grid cell modules to counteract 
  the increased projection errors from highercells per module. The experiment 
  was repeated 10 times, with the 5th, 50th, and 95th percentiles shown.
  """
  capacity_test.runExperiment()


if __name__ == "__main__":

  # Map paper figures to experiment
  generateFigureFunc = {
    "5A": generateFigure5A,
    "5B": generateFigure5B,
    "6": generateFigure6,
    "7": generateFigure7,
    "8": generateFigure8,
 }
  figures = generateFigureFunc.keys()

  parser = argparse.ArgumentParser(
    description="Use this script to generate the figures and results",
    epilog='''
    Kalvyn Roux, Dawie van den Heever.
    Orientation Invariant Sensorimotor Object Recognition Using Cortical 
    Grid Cells
    ''')

  parser.add_argument(
    "figure",
    metavar="FIGURE",
    nargs='?',
    type=str,
    default=None,
    choices=figures,
    help=("Specify the figure name to generate. Possible values are: %s " % figures)
  )
  parser.add_argument(
    "-c", "--cpuCount",
    default=None,
    type=int,
    metavar="NUM",
    help="Limit number of cpu cores.  Defaults to all available cores"
  )
  parser.add_argument(
    "-l", "--list",
    action='store_true',
    help='List all figures'
  )
  opts = parser.parse_args()

  if opts.list:
    for fig in generateFigureFunc:
      print(fig, generateFigureFunc[fig].__doc__)
  elif opts.figure is not None:
    generateFigureFunc[opts.figure](
      cpuCount=opts.cpuCount)
  else:
    parser.print_help()