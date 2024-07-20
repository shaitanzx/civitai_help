import os
import ssl
import sys



root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

from launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content


torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch torchvision --index-url {torch_index_url}")
requirements_file = os.environ.get('REQS_FILE', "requirements.txt")

run_pip(f"install -r \"{requirements_file}\"", "requirements")

from gradio_demo import *
