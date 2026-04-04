from setuptools import setup, find_packages

setup(
    name="src",        # The name people will use to install it
    version="0.1.0",
    packages=find_packages(),      # Automatically finds "my_package"
    # install_requires=[             # List external dependencies here
        # "requests>=2.25.1",
    # ],
    author="Aswathy Ajith",
    description="Counting behavior in LLMs",
)