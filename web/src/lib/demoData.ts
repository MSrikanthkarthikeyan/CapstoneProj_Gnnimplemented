import { KnowledgeGraph, LessonQuiz } from "./api";

export const DEMO_GRAPH: KnowledgeGraph = {
  goal: "Master Graph Neural Networks (GNNs)",
  nodes: [
    {
      id: "node_1",
      title: "1. Graph Theory Foundations",
      description: "Nodes, edges, adjacency matrices, degree matrices, and graph representations.",
      difficulty: 1,
      source_url: "https://en.wikipedia.org/wiki/Graph_theory",
    },
    {
      id: "node_2",
      title: "2. Vector Embeddings & Node Features",
      description: "Representing continuous node feature spaces and input matrices X in R^(N x D).",
      difficulty: 2,
      source_url: "https://en.wikipedia.org/wiki/Feature_(machine_learning)",
    },
    {
      id: "node_3",
      title: "3. Spatial Message Passing Framework",
      description: "Neighborhood aggregation, message transformation, and permutation-invariant update functions.",
      difficulty: 2,
      source_url: "https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/",
    },
    {
      id: "node_4",
      title: "4. Graph Convolutional Networks (GCN)",
      description: "Spectral approximations, normalized symmetric adjacency laplacians, and Kipf-Welling layers.",
      difficulty: 3,
      source_url: "https://www.youtube.com/watch?v=2KRAO4UecDU",
    },
    {
      id: "node_5",
      title: "5. Graph Attention Networks (GAT)",
      description: "Self-attention coefficients, dynamic neighbor weighting, and multi-head attention over graphs.",
      difficulty: 3,
      source_url: "https://en.wikipedia.org/wiki/Graph_neural_network",
    },
    {
      id: "node_6",
      title: "6. Over-Smoothing & Deep GNNs",
      description: "Representation collapse under repeated aggregation, Dirichlet energy decay, and residual solutions.",
      difficulty: 4,
      source_url: "https://ocw.mit.edu/",
    },
    {
      id: "node_7",
      title: "7. Inductive Learning with GraphSAGE",
      description: "Fixed-size neighborhood sampling, pooling aggregators, and scaling to massive web graphs.",
      difficulty: 4,
      source_url: "https://www.youtube.com/watch?v=JtDgmmQ60x8",
    },
    {
      id: "node_8",
      title: "8. Link Prediction & Recommender GNNs",
      description: "Edge classification, Bayesian Personalized Ranking (BPR) loss, and prerequisite-aware learning paths.",
      difficulty: 5,
      source_url: "https://en.wikipedia.org/wiki/Recommender_system",
    },
  ],
  edges: [
    { source: "node_1", target: "node_2", relation: "prerequisite" },
    { source: "node_2", target: "node_3", relation: "prerequisite" },
    { source: "node_3", target: "node_4", relation: "prerequisite" },
    { source: "node_3", target: "node_5", relation: "prerequisite" },
    { source: "node_4", target: "node_6", relation: "prerequisite" },
    { source: "node_4", target: "node_7", relation: "prerequisite" },
    { source: "node_5", target: "node_7", relation: "prerequisite" },
    { source: "node_7", target: "node_8", relation: "prerequisite" },
  ],
  mermaid: "graph TD\nnode_1 --> node_2\nnode_2 --> node_3\nnode_3 --> node_4\nnode_3 --> node_5\nnode_4 --> node_6\nnode_4 --> node_7\nnode_5 --> node_7\nnode_7 --> node_8",
  progress: {
    node_status: {
      node_1: "mastered",
      node_2: "mastered",
      node_3: "in_progress",
      node_4: "available",
      node_5: "locked",
      node_6: "locked",
      node_7: "locked",
      node_8: "locked",
    },
    node_scores: {
      node_1: 0.95,
      node_2: 0.88,
    },
    current_node: "node_3",
  },
};

export const DEMO_LESSONS: Record<string, LessonQuiz> = {
  node_3: {
    done: false,
    node_id: "node_3",
    node_title: "3. Spatial Message Passing Framework",
    node_description: "Neighborhood aggregation, message transformation, and permutation-invariant update functions.",
    difficulty: 2,
    source_url: "https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/",
    lesson: `In modern Graph Neural Networks (GNNs), the spatial message-passing paradigm serves as the universal foundational blueprint.

Unlike standard convolutional neural networks that operate on rigid Cartesian grids (such as 2D pixels), graphs possess irregular, non-Euclidean topologies with arbitrary node degrees and permutations.

At each layer k, every node v undergoes two essential transformations:
1. Message Aggregation: Node v gathers feature representations from its direct 1-hop topological neighbors N(v) using a symmetric, permutation-invariant operator (e.g., sum, mean, or max pooling).
2. State Update: The aggregated message vector is combined with node v's prior representation h_v^(k-1) and projected through non-linear learnable weights:
   h_v^(k) = UPDATE( h_v^(k-1), AGGREGATE({ MESSAGE(h_u^(k-1)) : u in N(v) }) )

By stacking K consecutive message-passing layers, each node's receptive field expands to encompass all information within its K-hop computational graph neighborhood.`,
    quiz: [
      {
        question: "Why must the neighborhood AGGREGATE function in a spatial GNN be permutation invariant?",
        options: [
          "Because graph nodes have no intrinsic canonical ordering, so permuting neighbor indices must not change output embeddings.",
          "Because neural networks can only execute mathematical operations on sorted vectors.",
          "Because permutation invariance guarantees that the adjacency matrix is strictly diagonal.",
          "Because it forces all node feature dimensions to reduce to 1.",
        ],
      },
      {
        question: "After stacking 3 consecutive message-passing layers, what is the effective receptive field of a node?",
        options: [
          "Its direct 1-hop immediate neighbors only.",
          "All nodes located within a 3-hop topological distance in the computational graph.",
          "The entire graph regardless of size or diameter.",
          "Only nodes that possess identical degree centrality.",
        ],
      },
    ],
  },
  node_4: {
    done: false,
    node_id: "node_4",
    node_title: "4. Graph Convolutional Networks (GCN)",
    node_description: "Spectral approximations, normalized symmetric adjacency laplacians, and Kipf-Welling layers.",
    difficulty: 3,
    source_url: "https://www.youtube.com/watch?v=2KRAO4UecDU",
    lesson: `Graph Convolutional Networks (Kipf & Welling, 2017) derive a first-order localized approximation of spectral graph convolutions.

A direct problem with naïve aggregation (A * H) is that multiplying by the raw adjacency matrix scales feature magnitudes by node degree, causing numerical instability and exploding gradients.

To solve this, GCN applies symmetric normalization combined with self-loops:
  H^(k+1) = sigma( D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2) * H^(k) * W^(k) )

Where:
- A_tilde = A + I_N (adding an identity matrix to preserve self-node features)
- D_tilde is the diagonal degree matrix corresponding to A_tilde
- W^(k) is the learnable layer weight matrix
- sigma is an element-wise activation function such as ReLU.`,
    quiz: [
      {
        question: "What is the primary role of adding self-loops (A + I) in the GCN layer formulation?",
        options: [
          "It ensures the node retains its own features during aggregation, rather than only averaging its neighbors.",
          "It forces the graph to become bipartite.",
          "It guarantees that all eigenvalues of the graph Laplacian become zero.",
          "It reduces the memory footprint of the sparse adjacency matrix.",
        ],
      },
    ],
  },
};
