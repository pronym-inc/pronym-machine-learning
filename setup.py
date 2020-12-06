from setuptools import find_packages, setup

install_dependencies = [
    'mypy==0.790',
    'numpy==1.19.2',
    'pandas==1.1.2',
    'scikit-learn==0.23.2'
]
test_dependencies = []


setup(
    name='pronym_machine_learning',
    url='https://github.com/pronym-inc/pronym-machine-learning',
    author='Pronym',
    author_email='gregg@pronym.com',
    entry_points={},
    packages=find_packages(),
    install_requires=install_dependencies,
    tests_require=test_dependencies,
    extras_require={'test': test_dependencies},
    include_package_data=True,
    version='0.1',
    license='MIT',
    description=('Some helpful description'),
    long_description=open('README.md').read(),
)