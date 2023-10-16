# CITS3001 Mario AI

## Authors
Robert Beashel 23489302
Martin Evans 23621647

## Description
This agent is a MCTS agent designed to function in the openAI gym environment gym-super-mario-bros.

## Environment Installation

A large number of packages for this installation, so please create a virtual env of your choice and run the following command


```bash
pip3 install -r requirements.txt
```

Also see: https://docs.python.org/3/library/venv.html, https://docs.conda.io/en/latest/

## Usage

The best way to use the agent is through invoking main.py, run in the following way:
Ensure your virtual environment is active when doing so

```bash
python3 main.py render_mode
```

With render_mode representing the display method of the mario environment. For instance:

```bash
python3 main.py human
```
