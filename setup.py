from setuptools import setup, find_packages

"""
Setup script for rapter-eggs.
"""

setup(
    name='rapter_eggs',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/espottesmith/RAPTER-eggs',
    author='Evan Walter Clark Spotte-Smith',
    author_email='espottesmith@gmail.com',
    python_requires=">=3.10",
    description="""
    Scripts to construct and analyze the RAPTER (ReActants, Products, and Transition-states of Elementary Reactions) and associated RAPTER-tracks and RAPTER-kettle datasets.
    """,
    keywords=[
        "schrodinger", "jaguar", "autots", "chemistry", "kinetics", "electronic", "structure", "transition-state"
    ]
)
