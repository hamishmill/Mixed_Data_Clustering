import os
import re
import ast
import pandas as pd
from pathlib import Path

from utils.plot_utils import plotXD

def generate_plots():

    base_path = Path(__file__).parent
    plots_folder = 'plots'
    output_folder = 'output'
    all_files = os.listdir(output_folder)

    for filename in all_files:
        print(f"Plotting: {filename}")
        filename = os.path.join(base_path, output_folder, filename)

        df = pd.read_csv(f"{filename}")

        plot_filename = Path(filename).stem
        plotfilepath = os.path.join(base_path, plots_folder, f"{plot_filename}_3D.pdf")

        # Colour is stored as a string representation of a list
        # but need a list so convert to list
        df['Colour'] = df.Colour.apply(lambda s: re.sub(r'\s+', ',', s))
        df['Colour'] = df.Colour.apply(lambda s: list(ast.literal_eval(re.sub(r'\s+', ',', s))))
        plotXD( df
                , title=''
                , XYZ=[0, 1, 2]
                , colour_column='Cluster'
                , size=100
                , save_to_file=plotfilepath
               )

        plotfilepath = os.path.join(base_path, plots_folder, f"{plot_filename}_2D.pdf")
        plotXD( df
                , title=''
                , colour_column='Cluster'
                , size=150
                , project2D = True
                , save_to_file=plotfilepath
               )
