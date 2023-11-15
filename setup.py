from setuptools import find_packages, setup


def get_long_description():
    return """This package implements EPoCS - an ESM-based Pocket Cross-Similarity metric for the comparison and contextualisation of protein binding sites, and systematic debiasing of train-test splits for pocket-centric machine-learning models. EPoCS combines protein language models (specifically, ESM-2) with real-space tesselation to generate vector embeddings for protein binding sites. The embeddings are the basis of the EPoCS similarity metric that gives rise to the pocket atlas."""  # noqa


def get_scripts():
    return [
        "./run_epocs.py",
    ]


if __name__ == "__main__":
    setup(
        name="epocs",
        version="0.1.0",
        url="https://github.com/toruc/epocs",
        description="EPoCS metric for binding-site similarity",
        long_description=get_long_description(),
        packages=find_packages(),
        scripts=get_scripts(),
        setup_requires=[],
        install_requires=[],
        include_package_data=True,
        ext_modules=[],
        license="Apache License 2.0",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        keywords="drug discovery machine learning protein binding sites similarity metric",
        python_requires=">=3.9",
    )
