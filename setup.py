import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='keyphrase_vectorizers',
    version='0.0.1',
    url='https://github.com/TimSchopf/Keyphrase_Vectorizers',
    license='BSD 3-Clause "New" or "Revised" License',
    author='Tim Schopf',
    author_email='tim.schopf@t-online.de.de',
    description='Set of vectorizers that extract keyphrases with part-of-speech patterns from a collection of text documents and convert them into a document-keyphrase matrix.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy >= 1.18.5',
        'spacy >= 3.0.1',
        'nltk >= 3.6.1',
        'scikit-learn >= 1.0'
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires='>=3.7',
)