from setuptools import setup, find_packages

setup(
    name='stock_prediction',
    version='0.1.0',
    description='AI-powered Stock Price Prediction Model',
    author='AI Engineering Team',
    author_email='ai_team@example.com',
    packages=find_packages(exclude=['tests', 'notebooks']),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'transformers>=4.12.0',
        'mlflow>=1.20.0',
        'optuna>=2.10.0',
        'yfinance>=0.1.70',
        'plotly>=5.3.0',
        'pyarrow>=5.0.0',
        'tqdm>=4.62.0',
        'pyyaml>=5.4.0'
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'flake8>=3.9.0',
            'mypy>=0.910',
            'black>=21.6b0'
        ],
        'gpu': [
            'cuda-python>=11.0',
            'nvidia-cudnn>=8.0'
        ]
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    entry_points={
        'console_scripts': [
            'stock_predict=src.predict:main',
            'stock_train=src.train:main'
        ]
    },
    package_data={
        'stock_prediction': ['config/*.yaml', 'models/*.h5']
    }
)