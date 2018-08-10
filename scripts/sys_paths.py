
# coding: utf-8

import os
import platform

platform = platform.system()

default_dir = os.getcwd()

if platform == 'Windows':

    scripts_dir = default_dir + '\scripts'
    connectome_data_dir = default_dir + '\connectome_data'
    inputmat_dir = default_dir + '\presets_input'
    voltagemat_dir = default_dir + '\presets_voltage'


else:

    scripts_dir = default_dir + '/scripts'
    connectome_data_dir = default_dir + '/connectome_data'
    inputmat_dir = default_dir + '\presets_input'
    voltagemat_dir = default_dir + '\presets_voltage'

