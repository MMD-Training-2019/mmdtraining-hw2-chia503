# Usage

Install Pandoc to compile `report.pdf`:

```sh
pip install -r requirements-doc.txt
# From report.md to report.pdf
./compile.sh
```

Without shell script on Windows (just copy and past):

```batch
> pip install -r requirements-doc.txt
> pandoc report.md -o report.pdf -M link-citations=true --bibliography=refer.bib --csl=ieee.csl --pdf-engine=xelatex -V CJKmainfont=DFKai-SB -V monofont=Consolas
```

# Links

PPT: <https://docs.google.com/presentation/d/1dQVeHfIfUUWxMSg58frKBBeg2OD4N7SD0YP3LYMM7AA/edit?usp=sharing>

Eample code: <https://bit.ly/32D5h6B>

Kaggle: <https://www.kaggle.com/c/ml2020spring-hw2>
