# coursework_dna_llm

This repository contains the research project "Identifying Biological Relations in the Human Genome Through Interpretation of Language Models", which focuses on analyzing DNA structures using deep learning and natural language processing techniques.

In this study, I evaluated the performance (fine-tuned) of the HyenaDNA model[^1] on specialized DNA structures, including G-quadruplexes (G4) and Z-DNA, and assessed its capability to identify promoter regions. Additionally, I used Explainable AI (XAI) methods to interpret the model's predictions and validate their biological relevance. To check it out in detail, head over to this repo!

## Project Structure

- `notebooks/` — Jupyter notebooks with experiments and data analysis.
- `src/` — Source code for models and utility functions.
- `data/` — Data and preprocessed files.
- `requirements.txt` — List of required dependencies.
- `report/` — Report and presentation of the project.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/perfectteatimer/coursework_zdna_llm.git
   cd coursework_zdna_llm
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ``` 
3. Launch Jupyter Notebook to reproduce experiments::
   ```bash
   jupyter notebook
   ``` 
## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to all contributors and the open-source community, epsecially to the authors of HyenaDNA for their groundbreaking work in long-range genomic sequence modeling.
- This project is inspired by the need to explore DNA tasks that can find applications in our lives.

[^1]: Nguyen, E., Poli, M., Faizi, M., et al. (2023). *HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution*. arXiv:2306.15794. [Link](https://arxiv.org/abs/2306.15794)
