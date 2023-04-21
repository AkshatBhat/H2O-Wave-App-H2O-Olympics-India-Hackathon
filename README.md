# Check out the Web App deployed at:
https://air-foresight-h2o-wave-app.onrender.com/

# Setting up H2O Wave locally

This project was bootstrapped with `wave init` command.

## Running the app

Make sure you have activated a Python virtual environment with `h2o-wave` installed.

If you haven't created a python env yet, simply run the following command (assuming Python 3.7 is installed properly).

For MacOS / Linux:

```sh
python3 -m venv venv
source venv/bin/activate
pip install h2o-wave
```

For Windows:

```sh
python3 -m venv venv
venv\Scripts\activate
pip install h2o-wave
```

Once the virtual environment is setup and active, run:

```sh
wave run app.py
```

Which will start a Wave app at <http://localhost:10101>.

## Learn More

To learn more about H2O Wave, check out the [docs](https://wave.h2o.ai/).
