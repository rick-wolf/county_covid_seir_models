import os
import yaml
import pandas as pd
from pyper import *
from string import Template

THIS_FILE_PATH = os.path.dirname(os.path.realpath('__file__'))

def get_age_distribution():
    return None

def extract_contact_matrices(config):
    """

    """

    r_script_path = os.path.join(THIS_FILE_PATH, 'contact_matrices.r')
    r_script = Template(open(r_script_path, 'r').read()).substitute(config['r_substitution'])
    r = R()
    r.run(r_script)

    return r.mr, r.m['participants']


