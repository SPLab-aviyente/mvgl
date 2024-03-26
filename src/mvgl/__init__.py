from mvgl.data import gen_simulated_data
from mvgl.graphlearning.multiview import learn_multiview_graph

# read version from installed package
from importlib.metadata import version
__version__ = version("mvgl")

# path to the project directory
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent.absolute().__str__()
