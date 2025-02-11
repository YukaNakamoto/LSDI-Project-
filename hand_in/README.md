# BTW 2025 Data Science Challenge Participation TU Berlin

## Introduction
This repository contains our solution for the BTW 2025 Data Science Challenge. The project leverages Python and Poetry for dependency management, ensuring a reproducible and consistent development environment.

## Getting Started

To set up the project locally, follow these steps:

### Prerequisites
Ensure you have the following installed on your system:
- Python (>=3.8)
- Poetry (>=1.0.0)

To install Poetry, run:
```sh
curl -sSL https://install.python-poetry.org | python3 -
```
Alternatively, on macOS, install it via Homebrew:
```sh
brew install poetry
```

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/BTW-2025-Data-Science-Challenge.git
   cd BTW-2025-Data-Science-Challenge
   ```

2. Install dependencies:
   ```sh
   poetry install
   ```
   This will create a virtual environment and install all dependencies specified in `pyproject.toml`.

3. Activate the virtual environment:
   ```sh
   poetry shell
   ```
   Now you are inside the virtual environment and can run Python scripts with the installed dependencies.

### Verify Dependencies
To check installed dependencies, run:
```sh
poetry show
```

### Running the Project
Run your main script using:
```sh
poetry run python your_script.py
```

### Running Jupyter Notebook with Poetry
To use Jupyter Notebook with Poetry dependencies, follow these steps:

1. Install Jupyter if not already installed:
   ```sh
   poetry add jupyter ipykernel
   ```

2. Add the Poetry environment as a Jupyter kernel:
   ```sh
   poetry run python -m ipykernel install --user --name=poetry-env --display-name "Python (Poetry)"
   ```

3. Start Jupyter Notebook:
   ```sh
   poetry run jupyter notebook
   ```

4. In Jupyter, select the kernel named `Python (Poetry)` to ensure the notebook runs with the Poetry dependencies.

### Deactivating the Virtual Environment
Exit the Poetry virtual environment by typing:
```sh
exit
```

## Contributing
If you wish to contribute, follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
For any inquiries, please contact `your-email@domain.com`.

