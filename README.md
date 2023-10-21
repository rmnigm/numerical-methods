# numerical-methods
![build status](https://github.com/rmnigm/numerical-methods/actions/workflows/ci.yml/badge.svg)
## What is it
This is a repository containing numerical algorithms, written in Python and C++.
They are used in homework for Numerical Methods course in 3rd year of Applied Math undergraduate program in HSE University. Homeworks are mostly annotated in russian.

**Homework topics:**
1. Numerical precision, computer representation of numbers
2. Solving nonlinear equations
3. Numerical precision for matrix operations
4. Solving systems of nonlinear equations
5. Solving systems of linear equations with iterative methods
6. Function approximation
7. Solving ordinary differential equations
8. Skipped that one.
9. Function optimization
10. Solving partial differential equations

## How to setup (Unix)
- Clone repository to local machine
- Install Pyenv using [this guide](https://github.com/pyenv/pyenv#installation) and install [Poetry](https://python-poetry.org)
- Install Python, used in project
  ```bash
  $ pyenv install 3.10.6
  ```
  If any problems happen - this [guide](https://github.com/pyenv/pyenv/wiki/Common-build-problems) can help.
- Create virtual environment with Poetry and install requirements:
  ```bash
  $ cd <path to cloned repo>
  $ poetry install
  ```

- Use Poetry tools for running scripts and testing:
  ```bash
  $ poetry run python <script>.py
  ```
And you are perfect, congratulations!

## C++ files building

Right now most of the cpp files are easily compilable without any need to start a project, if you have G++ installed.
```bash
  $ g++ <file>.cpp -o <executable_name>
  $ sh ./<executable_name>
  ```
