from setuptools import setup, find_packages

setup(
    name='ira_common',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python==4.9.0',
        'matplotlib',
        'scipy',
        'face-recognition==1.3.0',
        'mediapipe==0.10.11',
        'scikit-image',
        'ultralytics==8.2.74',
        'openai'
    ],
    author='Emma Brass',
    author_email='emma@brassville.com',
    description='Code for reuse amoungst all IRA projects.',
    url='https://github.com/yourusername/base_code',  # Optional: if hosted on GitHub #TODO put on github
)