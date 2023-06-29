# This script runs the variant effect prediction experiments.
# For each model, we extract the cosine distance between the reference and mutated sequence embedding.

# ResNet and AWD-LSTM.
python3 scripts/predict_variant_effects.py data/variant_effects/variant_effects.bed results/variant_effects_convnet.csv convnet pretrained_models/convnet data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256 
python3 scripts/predict_variant_effects.py data/variant_effects/variant_effects.bed results/variant_effects_awd_lstm.csv convnet pretrained_models/awd_lstm data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256

# Nucleotide Transformer. The correct kmer token is at position 43. After CLS removal, the position is 42.
python3 scripts/predict_variant_effects.py data/variant_effects/variant_effects.bed results/variant_effects_nt_1000g_500m.csv nt InstaDeepAI/nucleotide-transformer-500m-1000g  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 42

python3 scripts/predict_variant_effects.py data/variant_effects/variant_effects.bed results/variant_effects_nt_1000g_25b.csv nt InstaDeepAI/nucleotide-transformer-2.5b-1000g  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 42

python3 scripts/predict_variant_effects.py data/variant_effects/variant_effects.bed results/variant_effects_nt_ms_25b.csv  nt InstaDeepAI/nucleotide-transformer-2.5b-multi-species  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 42

python3 scripts/predict_variant_effects.py data/variant_effects/variant_effects.bed results/variant_effects_nt_href_500m.csv  nt InstaDeepAI/nucleotide-transformer-500m-human-ref  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 42


# From tokenizer: (add 1 for CLS)  (and subtract again if CLS is removed)
# 3: kmer 255 (centered)
# 4: kmer 255 (2nd position)
# 5: kmer 254 (centered)
# 6: kmer 254 (3rd position)

# DNABERT 3-mer: use token centered on variant. This is token 256, 255 after CLS removal.
python3 scripts/predict_variant_effects.py data/variant_effects/variant_effects.bed results/variant_effects_dnabert_3.csv dnabert pretrained_models/dnabert/3-new-12w-0 data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 255 --kmer 3
# DNABERT 4-mer: use token with variant at pos 2. This is token 256, 255 after CLS removal.
python3 scripts/predict_variant_effects.py data/variant_effects/variant_effects.bed results/variant_effects_dnabert_4.csv dnabert pretrained_models/dnabert/4-new-12w-0 data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 255 --kmer 4
# DNABERT 5-mer: use token centered on variant. This is token 255, 254 after CLS removal.
python3 scripts/predict_variant_effects.py data/variant_effects/variant_effects.bed results/variant_effects_dnabert_5.csv dnabert pretrained_models/dnabert/5-new-12w-0  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 254 --kmer 5
# DNABERT 6-mer: use token with variant at pos 2. This is token 255, 254 after CLS removal.
python3 scripts/predict_variant_effects.py data/variant_effects/variant_effects.bed results/variant_effects_dnabert_6.csv dnabert pretrained_models/dnabert/6-new-12w-0 data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 254 --kmer 6
