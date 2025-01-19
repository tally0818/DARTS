# DARTS
## 0. Introduction
The early stages of Neural Architecture Search (NAS) research focused on iteratively sampling architectures from the search space, training them, and using their performance to guide the search. The main drawback of these methods, when applied without speedup techniques (such as zero-cost proxy), is their immense computational cost. This is due to the need to train thousands of architects independently from scratch.
 As an alternative, one-shot techniques were introduced to train all architectures in the search space via a single training of a supernetwork. DARTS is an efficient one-shot technique that relaxes the search space to be continuous so that architecture can be optimized with respect to its validation set performance by gradient descent.
## 1. Method
1.1. Search space

Similar to the NASNet search space, we search for two types of computation cells: normal cells and reduction cells as the building blocks of the final architecture.
 A cell is a directed acyclic graph (DAG) consisting of an ordered sequence of N nodes. When the number N is given, it includes 2 input nodes, N-3 intermediate nodes, and 1 output node. For convolutional cells, the input nodes are defined as the cell outputs in the previous two layers. Each node $x(i)$ is a latent representation, and each directed edge $(i, j)$ is associated with some operation $o(i, j)$ that transforms $x(i)$. Each intermediate node is computed based on all of its predecessors:

$$x^{(i)}=\sum_{i<{j}}{o^{(i, j)}(x^{(i)})}^{}$$

The output cell is obtained by concatenating all the intermediate nodes.

1.2. Continuous relxations and optimization

 Let $O$ be a set of candidate operations where each operation represents some function $o()$ to be applied to $x(i)$. To make the search space continuous, we relax the categorical choice of a particular operation to a softmax over all possible operations on its edges:

 $$\overline{o}^{(i,j)}(x)=\sum_{o\in {O}}\frac{exp(\alpha_{o}^{(i,j)})}{\sum_{o'\in O}exp(\alpha_{o'}^{(i,j)}){}^{}}^{}o(x)$$

Since we're finding two types of cells (normal and reduction cells), the size of α would be $2 * number of edges * |O|$. When w represents the weight of each operation, our goal is to solve a bilevel optimization problem:

$$\mathscr{L}_{val}(w^{*}(\alpha),\alpha)\\w^{*}(\alpha)=\min_{w}\mathscr{L}_{train}(w,\alpha)$$

1.3. Approximate archtecture gradient

Since evaluating the architecture gradient exactly can be prohibitive due to the expensive inner optimization, the authors proposed a simple approximation scheme:
Using single-step approximation, we can approximate $\nabla _{\alpha}\mathscr{L}_{val}(w^{*}(\alpha),\alpha)$ as 

$$\nabla _{\alpha}\mathscr{L}_{val}(w^{*}(\alpha),\alpha)\approx\nabla _{\alpha}\mathscr{L}_{val}(w-\xi\nabla _{w}\mathscr{L}_{train}(w,\alpha),\alpha)$$

let $w'=w-\xi\nabla _{w}\mathscr{L}_{train}(w.\alpha)$ then by chain-rule,

$$\nabla_{\alpha}\mathscr{L}_{val}(w-\xi\nabla _{w}\mathscr{L}_{train}(w.\alpha),\alpha)=\nabla_{\alpha}\mathscr{L}_{val}(w',\alpha)-\xi\nabla _{\alpha,w}^{2}\mathscr{L}_{train}(w,\alpha)\nabla_{w'}\mathscr{L}_{val}(w',\alpha)$$

let $w^{\pm}=w\pm \varepsilon\nabla _{w'}\mathscr{L}_{val}(w',\alpha)$  then, by finite difference approximation,

$$_{\alpha,w}^{2}\mathscr{L}_{train}(w,\alpha)\nabla_{w'}\mathscr{L}_{val}(w',\alpha)\approx\frac{\nabla _{\alpha}\mathscr{L}_{train}(w^{+},\alpha)-\nabla _{\alpha}\mathscr{L}_{train}(w^{-},\alpha)}{2\epsilon}$$

By using this scheme, complexity is reduced from $O(|\alpha||w|)$ to $O(|\alpha|+|w|)$

1.4. Deriving discrete architectures

 To derive the final discrete convolutional cells, we retain the top-2 strongest operations among all non-zero candidate operations collected from all the previous nodes.
The strength of operation o is defined as:

$$\frac{exp(\alpha_{o}^{(i,j)})}{\sum_{o'\in O}^{}exp(\alpha_{o'}^{(i,j)})}$$

Zero operations are excluded to make a fair comparison with existing models (to match the number of operations) and because the strength of zero operations is underdetermined, as they only affect the scale of the resulting node representations while not affecting the final classification output.

## 2. Experiment
2.1. Searching for convolutional cells on CIFAR-10

 In my experiment, the operation set O included: 1×3 followed by 3×1 convolution, 3×3 average pooling, 3×3 max pooling, 3×3 convolution, 3×3 and 5×5 separable convolutions, 3×3 and 5×5 dilated convolutions, identity, and zero operations. All convolutional operations followed the ReLU-Conv-BN order. The convolutional cell consisted of N=7 nodes, with other details matching the NASNet search space.
During the search process, the macrostructure was fixed as [1,1,1] (comprising a stem layer, one normal cell, one reduction cell, one normal cell, and a classification layer). The initial number of channels for the stem layer was set to 8. The architecture parameters and model weights were optimized using ADAM and SGD respectively, following the cosine schedule as described in the reference paper.

## 3. Results
3.1. Founded cells
The cells I found by running my code on the setting of 2.1 are like the below:



## 4. Drawbacks
4.1. Rank disorder

 The rank disorder occurs when the ranking of architectures evaluated with the supernetwork doesn't match the ranking obtained from training them independently. Nearly all one-shot methods suffer from this problem. To overcome this issue approaches such as gradually increasing the network depth or gradually pruning the set of operation candidates during training have been proposed.

4.2. High memory consumption

 Since we're training a single supernetwork instead of thousands of smaller networks, training requires a massive amount of memory. It scales linearly with the size of the operation candidate set. To address this problem, methods such as ProxylessNAS and PC-DARTS have been proposed.

4.3. Operation biases

 While not observed in our results (possibly due to insufficient training time in the given 사지방 environment), several studies have shown that differentiable NAS techniques tend to favor skip connections over other operation choices. This might be caused by the supernetwork using skip connections to compensate for vanishing gradients. To address this bias, methods such as applying early stopping based on the stability of architecture weight rankings have been proposed. However, some researchers argue that favoring skip connections over other operation choices may be acceptable.

## 5. References
https://arxiv.org/pdf/1806.09055
https://arxiv.org/pdf/2301.08727
