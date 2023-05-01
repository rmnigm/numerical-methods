# numerical-methods
## What is it
This is a repository containing numerical algorithms, written in Python and C++.
They are used in homework for Numerical Methods course in 3rd year of Applied Math undergraduate program in HSE University. Homeworks are mostly annotated in russian.

**Homework topics:**
1. Numerical precision, computer representation of numbers
2. Solving nonlinear equations
3. Numerical precision for matrix operations
4. Solving systems of nonlinear equations
5. Solving systems of linear equations with iterative methods

## How to setup (Unix)
- Clone repository to local machine
- Install Pyenv using [this guide](https://github.com/pyenv/pyenv#installation)
- Install Python, used in project
  ```bash
  $ pyenv install 3.10.6
  ```
  If any problems happen - this [guide](https://github.com/pyenv/pyenv/wiki/Common-build-problems) can help.
- Create virtual environment for Python in repo
  ```bash
  $ cd <path to cloned repo>
  $ ~/.pyenv/versions/3.10.6/bin/python -m venv nums_env
  ```
- Activate venv (will be active until you clode the terminal session or use `deactivate`)
  ```bash
  $ source nums_env/bin/activate
  ```  
  In terminal you will now have a prefix:
  ```bash
  (nums_env)$ ...
  ```

- Check everything is correct and `python` and `pip` lead to `nums_env`
    ```bash
    (nums_env)$ which python
    <path to repo>/nums_env/bin/python
    (nums_env)$ which pip
    <path to repo>/nums_env/bin/pip
    ```
- Install dependencies using requirements.txt
  ```bash
  (nums_env)$ pip install --upgrade -r requirements.txt
  ```
And you are perfect, congratulations!

## C++ files building

Right now most of the cpp files are easily compilable without any need to start a project, if you have G++ installed.
```bash
  $ g++ <file>.cpp -o <executable_name>
  $ sh ./<executable_name>
  ```
