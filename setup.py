from setuptools import setup, find_packages

# Read requirements
def get_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="zero-harm-ai-detectors",
    packages=find_packages(),
    install_requires=get_requirements(),
    author="Zero Harm AI LLC",
    author_email="info@zeroharmai.com",
    description="Privacy and harmful content detection library",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Zero-Harm-AI-LLC/zero-harm-ai-detectors",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
)