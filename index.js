//readme: para executar esse programa é necessário usar a versão do node 22 e do tensorflow 4.22
//nvm install 22
//nvm use 22
//npm install @tensorflow/tfjs-node@4.22
//pode ocorrer um erro ao executar o programa por causa da DLL tensorflow.dll para corrigir execute o comando abaixo no powershell (como administrador) na pasta do projeto:
//Copy-Item ".\node_modules\@tensorflow\tfjs-node\deps\lib\tensorflow.dll" -Destination ".\node_modules\@tensorflow\tfjs-node\lib\napi-v8\tensorflow.dll"

import tf from '@tensorflow/tfjs-node';

//Essa função realiza o treinamento do modelo de rede neural com os dados de entrada (inputXs) e saída (outputYs).
async function trainModel(inputXs, outputYs) {
    //Esse objeto representa a arquitetura da rede neural, que é composta por camadas de neurônios. 
    // A função tf.sequential() cria um modelo sequencial, onde as camadas são empilhadas uma após a outra.
    //Ele funciona como uma sequencia de camadas, onde a saída de uma camada é a entrada da próxima. 
    // Isso permite que o modelo aprenda padrões complexos nos dados de entrada e faça previsões precisas.
    const model = tf.sequential()

    // Primeira camada da rede:
    // entrada de 7 posições (idade normalizada + 3 cores + 3 localizacoes)

    // 80 neuronios = aqui coloquei tudo isso, pq tem pouca base de treino
    // quanto mais neuronios, mais complexidade a rede pode aprender
    // e consequentemente, mais processamento ela vai usar

    // A ReLU age como um filtro:
    // É como se ela deixasse somente os dados interessantes seguirem viagem na rede
    /// Se a informação chegou nesse neuronio é positiva, passa para frente!
    // se for zero ou negativa, pode jogar fora, nao vai servir para nada
    // Essas são as camadas escondidas (hidden layers) da rede neural, que são responsáveis por aprender padrões complexos 
    // nos dados de entrada.
    // Aqui foram colocado 80 neurônios na primeira camada oculta, que é uma quantidade razoável para um conjunto de dados pequeno. 
    // A função de ativação ReLU (Rectified Linear Unit) é usada para introduzir não-linearidade na rede, permitindo que ela aprenda relações mais complexas entre os dados.
    // Esse dado de neurônios foi meio chutado de forma a verificar a acertividade do modelo, ou seja, 
    // foram sendo acrescentados neurônios até que o modelo conseguisse acertar a classificação das pessoas do dataset de treino.
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    // Saída: 3 neuronios
    // um para cada categoria (premium, medium, basic)

    // activation: softmax normaliza a saida em probabilidades
    // Essa camada de saída da rede neural é responsável por gerar as previsões finais do modelo. 
    // A função de ativação softmax é usada para normalizar os scores de cada categoria em probabilidades, 
    // garantindo que a soma das probabilidades seja igual a 1.
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    // Compilando o modelo 
    // Primeiro alimentamos a rede com os dados e camadas de entrada e saída.
    // Agora precisamos definir como o modelo vai aprender com os dados, ou seja, 
    // qual algoritmo de otimização e função de perda serão usados.
    // optimizer Adam ( Adaptive Moment Estimation)
    // é um treinador pessoal moderno para redes neurais:
    // ajusta os pesos de forma eficiente e inteligente
    // aprender com historico de erros e acertos

    // loss: categoricalCrossentropy
    // Ele compara o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa
    // a categoria premium será sempre [1, 0, 0]

    // quanto mais distante da previsão do modelo da resposta correta
    // maior o erro (loss)
    // Exemplo classico: classificação de imagens, recomendação, categorização de
    // usuário
    // qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    // Treinamento do modelo
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0, //informa se quer log do treinamento (0 = sem log, 1 = log detalhado, 2 = log resumido)
            epochs: 100, //quantidade de vezes que o modelo vai rodar no dataset
            shuffle: true, //embaralha os dados, para evitar viés (dados viciados)
            callbacks: {
                // onEpochEnd: (epoch, log) => console.log(
                //     `Epoch: ${epoch}: loss = ${log.loss}`
                // ) //Aqui está comentado pois só foi necessário na criação do código para verificar se o 
                //     modelo estava aprendendo corretamente. Depois de pronto não é necessário exibir o log de cada época, 
                // pois o modelo já está treinado e pronto para uso.
            }
        }
    )

    return model
}


// Essa função realiza a predição da categoria de uma nova pessoa usando o modelo treinado.
async function predict(model, pessoa) {
    // transformar o array js para o tensor (tfjs)
    const tfInput = tf.tensor2d(pessoa)

    // Faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}
// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
//Esses são os dados de entrada (features) que serão usados para treinar o modelo. 
// Cada linha representa uma pessoa, e cada coluna representa uma característica (idade normalizada, cor e localização). A normalização da idade é feita para que todos os valores fiquem na mesma escala, facilitando o aprendizado do modelo. As cores e localizações são representadas usando one-hot encoding, onde cada categoria é representada por um vetor binário.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded), dados de saída ou resultado esperado do modelo.
// São as respostas corretas para cada pessoa do dataset de treino.
// [premium, medium, basic]
// Cada linha representa a categoria correta para cada pessoa do dataset de treino. 
// A primeira linha corresponde a Erick, que é premium, a segunda linha corresponde a Ana, que é medium, 
// e a terceira linha corresponde a Carlos, que é basic. O modelo será treinado para aprender 
// a associar as características de cada pessoa com sua respectiva categoria.
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
// O TensorFlow.js trabalha com tensores, que são estruturas de dados multidimensionais. 
// Aqui, estamos convertendo os arrays JavaScript em tensores 2D, que serão usados como entrada (inputXs) e saída (outputYs) 
// para o treinamento do modelo. 
// O inputXs contém as características das pessoas, enquanto o outputYs contém as categorias correspondentes.
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// quanto mais dado melhor!
// assim o algoritmo consegue entender melhor os padrões complexos
// dos dados
// Aqui, estamos chamando a função trainModel para treinar o modelo de rede neural usando os dados de entrada (inputXs) e saída (outputYs).
const model = await trainModel(inputXs, outputYs)

//A partir de agora, podemos usar o modelo treinado para fazer previsões sobre novas pessoas.
//Aqui colocamos dados de uma nova pessoa para fazer a previsão de sua categoria (premium, medium ou basic).
//Não pode ser uma pessoa do dataset de treino, pois o modelo já conhece essas pessoas e não seria um teste válido.
const pessoa = { nome: 'zé', idade: 28, cor: 'verde', localizacao: "Curitiba" }
// Normalizando a idade da nova pessoa usando o mesmo padrão do treino
// Exemplo: idade_min = 25, idade_max = 40, então (28 - 25) / (40 - 25 ) = 0.2

// Criandos os dados normalizados e one-hot encoded para a nova pessoa, 
// seguindo a mesma ordem usada no treino: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        1,    // cor azul
        0,    // cor vermelho
        0,    // cor verde
        0,    // localização São Paulo
        1,    // localização Rio
        0     // localização Curitiba
    ]
]

// Fazendo a previsão da categoria da nova pessoa usando o modelo treinado
const predictions = await predict(model, pessoaTensorNormalizado)
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')
console.log(results)