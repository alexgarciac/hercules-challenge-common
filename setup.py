import setuptools

with open("README.md", 'r') as f:
      readme_contents = f.read()

setuptools.setup(name='herc_challenge_common',
      version='0.5.2',
      author="Alejandro González Hevia",
      author_email="alejandrgh11@gmail.com",
      description="",
      long_description=readme_contents,
      long_description_content_type="text/markdown",
      packages = setuptools.find_packages(),
      python_requires='>=3.6'
)
