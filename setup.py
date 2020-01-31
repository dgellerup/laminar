from setuptools import setup, find_packages
PACKAGES = find_packages()

opts = dict(name='laminar',
            maintainer='Dane Gellerup',
            maintainer_email='danegellerup@uwalumni.com',
            description='Simpler parallelization.',
            long_description=open('README.md').read(),
            long_description_content_type="text/markdown",
            url='https://github.com/dgellerup/laminar',
            download_url='https://github.com/dgellerup/laminar/archive/v1.1.5.tar.gz',
            keywords=['laminar', 'parallel', 'parallelization', 'parallel processing', 'processing'],
            license='MIT',
            author='Dane Gellerup',
            author_email='danegellerup@uwalumni.com',
            version='1.1.5',
            packages=PACKAGES,
            install_requires=["pandas>=0.24.0",
                                "numpy>=1.12.1"],
            classifiers=[
                'Development Status :: 5 - Production/Stable',
                'Intended Audience :: Developers',
                'Topic :: Software Development :: Build Tools',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.6',
                'Programming Language :: Python :: 3.7',
                'Programming Language :: Python :: 3.8',
            ]
           )

setup(**opts)
