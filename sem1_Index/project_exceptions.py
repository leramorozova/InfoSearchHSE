import os


class NoZipfile(Exception):
    def __init__(self, project_root):
        abs_path = os.path.abspath(project_root)
        msg = f"There is no corpus zipfile in {abs_path}. Please, put it in your project root directory."
        super().__init__(msg)
