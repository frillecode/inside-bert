<br />

  <h1 align="center">Investigating information dynamics in BERT models
during fine-tuning </h1> 

  <p align="center">
    Frida Hæstrup (201805753)
    <br>
</p>


## Repository structure

```bash
inside-bert/  
├── src/ 
│   └── configs/
│   │   └── *  # experiment-specific-configs
│   └── utils/
│   │   └── experiments.py # the class for creating an experiment
│   │   └── utils_finetune.py # methods for for fine-tuning
│   │   └── utils_infodynamics.py # methods for extracting information signals
│   │   └── utils_visualizations.py  # methods for visualizing results
│   └── main-finetune.py # main driver for fine-tuning
│   └── main-infodynamics.py # main driver for information dynamisc
```


## Technicalities
The codebase relies on python v 3.12.3.


## Licencse
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
