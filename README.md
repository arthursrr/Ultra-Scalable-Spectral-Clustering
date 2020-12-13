# Ultra Scalable Spectral Clustering

Inicialmente este trabalho foi realizado para a cadeira de Algebra Linear para Aprendizado de Máquina - PUC-Rio.

Este repositório apresenta uma implementação do Ultra Scalable Spectral Clustering.

## O que é?

Antes de iniciar, é preciso que você tenha algum conhecimento sobre **Espectral Clustering**

Você pode visualizar um notebook explicando seu funcionamento [aqui](Notebooks/Original_Spectral_Clustering.ipynb)

Para lidar com datasets (conjunto de dados) extremamente grandes, o algoritmo **U-SPEC** proposto usa uma abordagem baseada em sub-matrix e visa quebrar o gargalo de eficiência por meio de três fases. 

Você pode visualizar melhor o artigo [aqui](Publications/TKDE.2019.2903410.pdf)

## Notas

* ### Imports

    Os algoritmos neste reposiório necessitam das seguintes bibliotecas

    | Imports       |
    |-|
    | pandas       |
    | numpy        |
    | matplotlib   |
    | scikit-learn |
    | scipy        |
    | tensorflow   |
    | tqdm         |

## Notebooks

Aqui deve ser colocado uma referência aos notebooks

## Uso

Aqui deve ser colocado uma referência ao código fonte

## Exemplos

Aqui deve ser colocado alguns resultados visuais, como métricas, tempo de execução etc

## Observação

Para operar em em uma quantidade muito grande de elementos (ex: 1 Milhão) é necessário utilizar a multiplicação de matrizes espasas, no momento essa operação só está otimizada em espaço, em tempo ainda demora muito. Caso haja alguma atualização do tensorflow com otimizações de operações Sparse X Sparse esse algoritmo será atualizado  

