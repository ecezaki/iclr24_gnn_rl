# iclr24_gnn_rl


This is a sample code for the GRRS algorithm which combines GNN and DQN for session-based recommendation 
(with warm network initialization).
This code is written to work on the Goodreads dataset.
Files we require are: "ratings.dat" which contains the user id, item id, rating and timestamp for every interaction.

Procedure to run: 
1) python preprocess.py
2) python pretrain.py
3) python main.py


Note: This code requires PyTorch and torch-geometric to run.
