from setuptools import setup

setup(
    name='keras_video_object_detector',
    packages=['keras_video_object_detector'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
        'sklearn'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)