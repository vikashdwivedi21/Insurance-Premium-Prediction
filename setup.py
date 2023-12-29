from setuptools import find_packages, setup
from typing import List

hypen_e_dot = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n","") for req in requirements ]

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)


        return requirements


setup(

    name= 'InsurencePremiumPrediction',
    version='0.0.1',
    author= 'Vikash Dwivedi',
    author_email= 'mrvikashdwivedi021@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    packages= find_packages()

)