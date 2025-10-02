from model_context import model_context
from videodepthanything_demo_modified import run

def run_depth_model():
    with model_context('VideoDepthAnything'):
        run()