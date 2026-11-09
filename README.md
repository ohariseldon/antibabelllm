# antibabelllm

O lado oposto do [babelllm](https://github.com/ohariseldon/babelllm). O babelllm reescreve texto de LLM para apagar marcas e rastros. Este repositório tenta enxergar o texto de LLM mesmo depois disso. Funciona em português do Brasil e em inglês.

São duas peças:

1. **Detector sem chave.** Ele olha o texto e diz se parece IA, sem saber quem gerou. Mede estilo e ritmo, usa um vetor de sentido multilíngue e sinais de um modelo de linguagem pequeno. O treino é adversário: o detector também vê saídas do babelllm.
2. **Marca semântica com chave (o "selo").** Serve para quem gera o texto e quer prová-lo depois. A marca fica no que cada frase diz, não nas palavras. Por isso sobrevive a paráfrase, troca de ordem, fusão e divisão de frases. A detecção dá um p-valor exato.

A régua é sempre a mesma: quanto de IA é pego quando só 1% do texto humano é acusado (TPR a 1% de falso positivo), por idioma e por ataque. O limiar é escolhido em textos humanos separados e aplicado a outros. Um artigo e todas as reescritas dele ficam sempre do mesmo lado da divisão.

## O selo, em quatro passos

1. Cada frase vira um vetor de sentido `e` (modelo `paraphrase-multilingual-MiniLM-L12-v2`).
2. Tira-se a média do documento, `r = e − média`. Sai o tema comum e fica o que cada frase acrescenta. Depois o vetor é branqueado com `W = C^(-1/2)`, onde `C` foi estimada em texto humano público (`dados/branqueador_minilm.npy`).
3. A chave sorteia 128 eixos, metade positivos e metade negativos. A nota da frase é `q(r) = Σ s_j ⟨r, g_j⟩² / (m‖r‖²)`. Ela é alta quando a frase varia ao longo dos eixos positivos. É uma função suave do sentido e não depende da posição da frase.
4. **Marcar:** para cada frase, entre N versões candidatas, fica a de nota maior. **Detectar:** tira-se a média das notas e compara-se com 999 chaves-isca. Sem marca, a chave verdadeira é só mais uma chave sorteada, então o posto dela entre as iscas dá um p-valor exato.

Fusões e divisões são tratadas medindo em três escalas: orações, frases e pares de frases vizinhas. Vale o máximo das três, e as iscas passam pelo mesmo máximo.

**Calibração por chave.** O p-valor pelo posto é exato na média das chaves. Uma chave fixa, porém, pode ter um viés próprio num gênero de texto. A calibração mede média e desvio do z de cada chave (a verdadeira e as iscas, pelo mesmo caminho) em texto humano de referência. Com ela, o falso positivo de cada chave fica no nível do acaso; os números estão abaixo.

```python
from selo import Selo
s = Selo("minha-chave-secreta")
s.calibrar(textos_humanos_de_referencia)   # uma vez por chave; salvar_calibracao() guarda
esc = s.escolher([[frase, versao1, versao2, ...], ...])   # índice da versão a manter
r = s.detectar(texto)   # r.z, r.p, r.marcado (p ≤ 0,01)
```

`gera_marcado.py` mostra a marca aplicada na hora de gerar, com um modelo local. Frase a frase, o modelo sorteia N continuações e o selo fica com a de nota maior.

## Resultados até agora (09/11/2026)

### Detector sem chave, pt-BR

- **Humanos:** colunas e notícias da Folha (2015-2017).
- **IA:** artigos do wasemu escritos pelo Claude e 20 desses artigos reescritos pelo babelllm (Haiku 4.5 e Gemini, várias configurações).
- **Divisão:** 5 dobras por artigo. O teste é sempre feito em artigos cujas reescritas o detector nunca viu.

| sinais | treino sem babelllm | treino com babelllm de outros artigos |
|---|---|---|
| só estilo | 30% (modo `prosa`, padrão atual: 11%) | 91% |
| estilo + vetor de sentido | 91% | 99% |

Os artigos do wasemu sem reescrita são pegos em 100% dos casos. O falso positivo real nos humanos de teste ficou entre 1,7% e 2,4%, para um nominal de 1%.

**Controle de assunto e época** (`exp_negativos.py`). Humanos e IA da tabela acima diferem também em época e assunto. Por isso a conta foi refeita com 506 notícias humanas de economia da Agência Brasil, de 2025-2026 e sobre os mesmos temas (licença CC BY).

- O detector da tabela acusava 12% delas.
- Com metade delas no treino como humano, o falso positivo volta a 1,5%. O detector que nunca viu babelllm cai de 91% para 73%; no modo prosa, de 87% para 57%.
- Com treino adversário, continua em 99%.

Ou seja, uns 18 pontos eram assunto e época; o resto é sinal.

**Ataque contra defesa** (`exp_ataque_novo.py <pasta> <nome>`). A thread do babelllm manda cada versão nova do ataque, e ela é medida aqui. Todos os números abaixo têm controle de assunto.

| versão do babelllm | nunca viu babelllm | viu as versões antigas | retreinado com esta |
|---|---|---|---|
| `--notas prosa` | 57% | 100% | 100% |
| `--notas fatos` (v0.8) | 76% | 99% | 99% |
| `--notas fatos` (fatos2) | 73% | 100% | 100% |

No fatos2, só o estilo pega 22% sem ter visto o babelllm e 89% depois de ver as versões antigas. Só o vetor de sentido pega 71% e 99%. As versões novas são pegas por um detector que só conhecia as antigas, então a família inteira deixa uma assinatura no espaço de sentido.

O Binoculars com o par pequeno Qwen2.5 (0,5B e 1,5B) não separa os artigos do wasemu das colunas de economia da Folha. Pior: as saídas do babelllm parecem *mais* humanas para ele.

### Detector sem chave, inglês (RAID, notícias)

Aqui cada gerador fica de fora do treino e é testado como se fosse novo. Os ataques do RAID estão incluídos.

| gerador nunca visto | só estilo | estilo + vetor de sentido |
|---|---|---|
| chatgpt | 87% | 97% |
| llama-chat | 90% | 96% |
| mistral-chat | 84% | 91% |
| gpt4 | 64% | 88% |
| cohere-chat | 41% | 56% |

### Selo: falso positivo de uma chave fixa

Cada uma das mil chaves fez, por sua vez, o papel de chave verdadeira. O teste usou 250 textos humanos por idioma (`exp_selo_fpr2.py`). Só pelo acaso, com 250 textos, o percentil 99 seria cerca de 2,8% (no nível de 1%) e 8,2% (no nível de 5%).

| versão | pt: média / p99 (nível 1%) | pt: média / p99 (nível 5%) | en: p99 (nível 5%) |
|---|---|---|---|
| sem correção | 1,0% / 4,4% | 5,0% / 14,8% | 13,2% |
| branqueado | 1,0% / 2,8% | 5,0% / 9,2% | 9,6% |
| branqueado + calibrado | 1,0% / 2,8% | 5,0% / 8,8% | 9,2% |

Sem correção, algumas chaves acusavam quase três vezes o nominal. Com branqueamento e calibração, a variação entre chaves fica no nível do acaso.

### Selo: força da marca

- **Escolha entre reescritas prontas do Haiku** (`exp_selo_pool.py`, cerca de 3,6 versões por frase, 20 artigos). O z médio do texto marcado foi 2,35, contra 0,86 no controle. Ele continuou entre 2,2 e 2,5 depois de embaralhar, fundir 70% das frases, dividir nas vírgulas e cortar 30% das frases. A escolha aqui é só entre palavras, então a força é pequena, e uma paráfrase nova tende a apagá-la.
- **Marca na hora de gerar** (`gera_marcado.py`, Qwen2.5-1.5B local, 8 continuações por frase). As continuações dizem coisas diferentes, então a marca fica no conteúdo. Num texto de 22 frases, o marcado teve z = 4,9 (p = 0,001), o controle −0,3 e o marcado depois de parafraseado 3,6 (p = 0,002). A rodada com 20 temas ainda está em andamento.

## Como rodar

```bash
pip install -r requirements.txt
export ANTIBABEL_TRABALHO=~/antibabel_trabalho     # ver config.py para as outras pastas
bash baixar_dados.sh "$ANTIBABEL_TRABALHO/data/raw"
python montar_dados.py
python extrair.py estilo && python extrair.py emb
python treina.py estilo emb --cv
python exp_selo_fpr.py --ajustar      # branqueador; depois exp_selo_fpr2.py
python -m unittest discover -s testes # testes rápidos, sem baixar modelo
```

## Referências

- Binoculars (2401.12070)
- Raidar (2401.12970)
- Krishna et al., defesa por recuperação (2303.13408)
- SemStamp (2310.03991) e k-SemStamp (2402.11399)
- SIR (2310.06356)
- RADAR (2307.03838)
- DAMAGE (2501.03437)
- AliMark (2605.29434)

Todos estão no arXiv.

A AliMark grava uma sequência de bits alinhada à ordem das frases. Por isso sofre com muitas fusões e divisões e com troca de ordem. O selo daqui não depende de ordem e é contínuo.
