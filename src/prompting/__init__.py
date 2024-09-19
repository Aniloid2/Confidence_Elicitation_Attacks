# import pkgutil
# import importlib
# import inspect
# import os

# def recursive_import(package_name, package_path):
#     # Iterate through all submodules in the package
#     for module_info in pkgutil.walk_packages(package_path, package_name + "."):
#         if not module_info.ispkg:
#             module = importlib.import_module(module_info.name)

#             # Import all public classes from the module
#             for name, obj in inspect.getmembers(module, inspect.isclass):
#                 if not name.startswith("_"):  # Exclude classes with names starting with '_'
#                     globals()[name] = obj

# # Current package name and path
# package_name = __name__
# package_path = [os.path.dirname(__file__)]

# # Import all classes recursively
# recursive_import(package_name, package_path)

from .prompt_config import DYNAMIC_PROMPT