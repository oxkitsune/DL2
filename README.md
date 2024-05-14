# DL2

Get the repo
1. !git clone https://github.com/ababier/open-kbp.git 'open-kbp'

2. Use python 3.10.10

3. pip install -r open-kbp/requirements.txt



### Snellius tutorial:
In order to run this on the snellius cluster, first add your SSH key to https:///portal.cua.surf.nl  


Then you can ssh into snellius, using
```sh
ssh scur0394@snellius.surf.nl
``` 


Once you're in the snellius box, you need to ssh into the interactive gpu workspace using:
```sh
ssh gcn1
```

On that machine you can use the venv under `~/.venv`, that comes with the Minkowski Engine pre-installed.

```sh
source ~/.venv/bin/activate
```

The repository is cloned under `~/DL2`.

Code can be run using the command
```
python -m src.main
```

