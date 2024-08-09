# Bayesian Combining


## Prerequisites after first pull

1. Create Python Virtual Environment:
    ```
    python -m venv env
    ```
2. Activate Virtual Environment:<br>
    **Linux**
    ```
    source env/bin/activate
    ```
    **Windows**
    ```
    env\Scripts\activate
    ```
3. Load Required libraries
    ```
    pip install -r requirements.txt
    ```
4. Give Script permissions in Linux:<br>
    ```
    chmod +x main.py
    ```
5. Deactivate Virtual Environment:<br>
    **Linux**
    ```
    source env/bin/deactivate
    ```
    **Windows**
    ```
    env\Scripts\deactivate
    ```
## Save new libraries installed in the requirements.txt
Inside the env:
```
python -m pip freeze > requirements.txt
```

## Normal Workflow to launch scripts:
1. Activate Virtual Environment:<br>
    **Linux**:
    ```
    source env/bin/activate
    ```
    **Windows**:
    ```
    env\Scripts\activate
    ```
2. Launch Python main **Serial Execution**:<br>
    ```
    python3 main.py
    ```
    **Launch Python main with Multiprocessing**:
    ```
    python3 main_multiprocessing.py
    ```
3. Deactivate Virtual Environment:<br>
    **Linux**
    ```
    source env/bin/deactivate
    ```
    **Windows**
    ```
    env\Scripts\deactivate
    ```
