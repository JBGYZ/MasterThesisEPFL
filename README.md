# MasterThesisEPFL
Sample complexity of transformer models on RHM

## Run
```bash
python main.py --pickle test --device cpu \
--dataset hier1 --m 8 --num_features 8 --num_layers 2 --ptr 0.4\
--net transformer --net_layers 1 --nhead 4 --dim_feedforward 256 \
--reducer_type fc --reducer_layers 1 --reducer_size 128 --width 1024 \
--epochs 1000 --scheduler none
```

## References

- Leonardo Petrini, Francesco Cagnetta, Umberto M. Tomasini, Alessandro Favero, Matthieu Wyart. [How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model](https://arxiv.org/abs/2307.02129). *arXiv*, 2023.

- Code largely inpired by [hierarchy-learning](https://github.com/pcsl-epfl/hierarchy-learning) repository.

