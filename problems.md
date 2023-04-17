# Problems

## Conda Usage

```python

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes

conda create --name myenv python=3.10

conda activate myenv

conda install numpy

conda deactivate

conda remove -n myenv --all

```

This is good