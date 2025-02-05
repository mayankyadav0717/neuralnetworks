# Neural Networks Learning Journey

This repository serves as a comprehensive learning log, documenting my progress as I follow [Andrej Karpathy's "Zero to Hero" playlist](https://www.youtube.com/playlist?list=PLpVmAiHaw0HOuHfh0eYSaVbcB6RCsMS9X). It contains Jupyter notebooks, personal handwritten notes, and practical implementations of fundamental and advanced neural network concepts.

## 📌 Contents

### Micrograd: A Minimalistic Autograd Engine
- **[micrograd.ipynb](https://github.com/mayankyadav0717/neuralnetworks/blob/main/micrograd.ipynb)**  
  Implementation of a simple yet powerful automatic differentiation engine from scratch, mimicking the core functionality of PyTorch’s autograd.  
  - **Key Concepts**: Backpropagation, computational graphs, autograd implementation.  
  - **Video Reference**: [Building Micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)

### Makemore: Character-Level Language Modeling
- **[makemore.ipynb](https://github.com/mayankyadav0717/neuralnetworks/blob/main/makemore.ipynb)**  
  Develops a simple bigram character-level language model, setting the foundation for more sophisticated deep learning models.  
  - **Key Concepts**: Bigram modeling, tokenization, `torch.Tensor` operations, loss functions, sampling techniques.  
  - **Video Reference**: [Bigram Model](https://www.youtube.com/watch?v=PaCmpygFfXo)

- **[makemore2.ipynb](https://github.com/mayankyadav0717/neuralnetworks/blob/main/makemore2.ipynb)**  
  Expands the model by implementing a multi-layer perceptron (MLP)-based character-level language model.  
  - **Key Concepts**: Model training workflow, hyperparameter tuning, train/dev/test splits, handling underfitting and overfitting.  
  - **Video Reference**: [MLP Model](https://www.youtube.com/watch?v=TCH_1BHY58I)

- **[makemore3.ipynb](https://github.com/mayankyadav0717/neuralnetworks/blob/main/makemore3.ipynb)**  
  Examines the inner workings of multilayer perceptrons and explores essential diagnostic techniques.  
  - **Key Concepts**: Activation statistics, weight initialization, batch normalization, gradient flow analysis.  
  - **Video Reference**: [Understanding MLP Internals](https://www.youtube.com/watch?v=P6sfmUTpUmc)

- **[makemore4.ipynb](https://github.com/mayankyadav0717/neuralnetworks/blob/main/makemore4.ipynb)**  
  Manually backpropagates through a 2-layer MLP with batch normalization without using PyTorch’s autograd. Provides a deep dive into gradient flow through various network components.  
  - **Key Concepts**: Manual backpropagation, cross-entropy loss, batch normalization, embeddings.  
  - **Video Reference**: [Manual Backpropagation](https://www.youtube.com/watch?v=q8SA3rM6ckI)

- **[makemore5.ipynb](https://github.com/mayankyadav0717/neuralnetworks/blob/main/makemore5.ipynb)**  
  Extends the previous 2-layer MLP into a hierarchical, tree-like architecture, resembling a WaveNet-style convolutional neural network.  
  - **Key Concepts**: Convolutional architectures, hierarchical model structures, deep learning development workflow.  
  - **Video Reference**: [Building a Deeper Model](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)

### GPT: Building Transformer-based Models
- **[gpt-dev.ipynb](https://github.com/mayankyadav0717/neuralnetworks/blob/main/gpt-dev.ipynb)**  
  Implements a Generatively Pretrained Transformer (GPT) following key principles outlined in "Attention is All You Need."  
  - **Key Concepts**: Transformer architecture, attention mechanisms, positional encodings, self-attention layers.  
  - **Video Reference**: [GPT Model](https://www.youtube.com/watch?v=kCc8FmEb1nY)

- **[gpttokenizer.ipynb](https://github.com/mayankyadav0717/neuralnetworks/blob/main/gpttokenizer.ipynb)**  
  Implements a Byte Pair Encoding (BPE) tokenizer used in large language models.  
  - **Key Concepts**: Tokenization, vocabulary construction, encoding and decoding, handling text chunks efficiently.  
  - **Video Reference**: [Understanding Tokenization](https://www.youtube.com/watch?v=zduSFxRajkE)

- **[gpt2](https://github.com/mayankyadav0717/neuralnetworks/tree/main/gpt2)**  
  Attempt at reproducing the full GPT-2 model from scratch, with limitations due to hardware constraints.  
  - **Key Concepts**: Large-scale model training, resource constraints, implementation challenges.  
  - **Video Reference**: [Reproducing GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU)

### 📖 Notes & Additional Resources
- **[Zero_to_hero.pdf](https://github.com/mayankyadav0717/neuralnetworks/blob/main/Zero_to_hero.pdf)**  
  A collection of handwritten notes summarizing key takeaways from the "Zero to Hero" series, including equations, intuitions, and insights into deep learning.

---

## 🚀 Getting Started
To explore the notebooks, clone this repository and open the Jupyter notebooks in your preferred environment:

```bash
git clone https://github.com/mayankyadav0717/neuralnetworks.git
cd neuralnetworks
jupyter notebook
```

---

## 🤝 Contributions & Feedback
This repository serves as a personal learning journey, but I welcome discussions, feedback, and suggestions! Feel free to open issues or reach out.

