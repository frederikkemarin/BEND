# This script runs the variant effect prediction experiments.
# For each model, we extract the cosine distance between the reference and mutated sequence embedding.

# Run expression variants.
OUT_DIR=results_variant_effects
VARIANT_FILE=data/variant_effects/variant_effects_expression.bed
OUT_PREFIX=variant_effects_expression
mkdir -p $OUT_DIR

python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_dnabert_6.csv dnabert pretrained_models/dnabert/6-new-12w-0 data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 254 --kmer 6 # index 254 has the variant at its 3rd position in the 6-mer
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_dnabert2.csv dnabert2 zhihan1996/DNABERT-2-117M  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_nt_1000g_500m.csv nt InstaDeepAI/nucleotide-transformer-500m-1000g  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_nt_1000g_25b.csv nt InstaDeepAI/nucleotide-transformer-2.5b-1000g  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_nt_ms_25b.csv nt InstaDeepAI/nucleotide-transformer-2.5b-multi-species  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_nt_href_500m.csv nt InstaDeepAI/nucleotide-transformer-500m-human-ref  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_hyenadna_tiny1k_512.csv hyenadna pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen ../GRCh38.primary_assembly.genome.fa --extra_context 512 --embedding_idx 511 # autoreregressive
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_hyenadna_small32k_512.csv hyenadna pretrained_models/hyenadna/hyenadna-small-32k-seqlen ../GRCh38.primary_assembly.genome.fa --extra_context 512 --embedding_idx 511 # autoreregressive
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_hyenadna_medium160k_512.csv hyenadna pretrained_models/hyenadna/hyenadna-medium-160k-seqlen ../GRCh38.primary_assembly.genome.fa --extra_context 512 --embedding_idx 511 # autoreregressive
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_hyenadna_medium450k_512.csv hyenadna pretrained_models/hyenadna/hyenadna-medium-450k-seqlen ../GRCh38.primary_assembly.genome.fa --extra_context 512 --embedding_idx 511 # autoreregressive
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_hyenadna_large1m_512.csv hyenadna pretrained_models/hyenadna/hyenadna-large-1m-seqlen ../GRCh38.primary_assembly.genome.fa --extra_context 512 --embedding_idx 511 # autoreregressive
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_genalm_bert_large_t2t.csv genalm AIRI-Institute/gena-lm-bert-large-t2t ../GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_genalm_bigbird_base_t2t.csv genalm AIRI-Institute/gena-lm-bigbird-base-t2t ../GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_convnet.csv convnet pretrained_models/convnet data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_awd_lstm.csv awdlstm pretrained_models/awd_lstm data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 511 --extra_context 512 #autoreregressive
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_nt_v2_500m.csv nt InstaDeepAI/nucleotide-transformer-v2-500m-multi-species data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_grover.csv grover pretrained_models/grover data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256


# Run disease variants.
OUT_DIR=results_variant_effects
VARIANT_FILE=data/variant_effects/variant_effects_disease.bed
OUT_PREFIX=variant_effects_disease
mkdir -p $OUT_DIR

python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_dnabert_6.csv dnabert pretrained_models/dnabert/6-new-12w-0 data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 254 --kmer 6 # index 254 has the variant at its 3rd position in the 6-mer
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_dnabert2.csv dnabert2 zhihan1996/DNABERT-2-117M  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_nt_1000g_500m.csv nt InstaDeepAI/nucleotide-transformer-500m-1000g  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_nt_1000g_25b.csv nt InstaDeepAI/nucleotide-transformer-2.5b-1000g  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_nt_ms_25b.csv nt InstaDeepAI/nucleotide-transformer-2.5b-multi-species  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_nt_href_500m.csv nt InstaDeepAI/nucleotide-transformer-500m-human-ref  data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_hyenadna_tiny1k_512.csv hyenadna pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen ../GRCh38.primary_assembly.genome.fa --extra_context 512 --embedding_idx 511 # autoreregressive
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_hyenadna_small32k_512.csv hyenadna pretrained_models/hyenadna/hyenadna-small-32k-seqlen ../GRCh38.primary_assembly.genome.fa --extra_context 512 --embedding_idx 511 # autoreregressive
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_hyenadna_medium160k_512.csv hyenadna pretrained_models/hyenadna/hyenadna-medium-160k-seqlen ../GRCh38.primary_assembly.genome.fa --extra_context 512 --embedding_idx 511 # autoreregressive
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_hyenadna_medium450k_512.csv hyenadna pretrained_models/hyenadna/hyenadna-medium-450k-seqlen ../GRCh38.primary_assembly.genome.fa --extra_context 512 --embedding_idx 511 # autoreregressive
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_hyenadna_large1m_512.csv hyenadna pretrained_models/hyenadna/hyenadna-large-1m-seqlen ../GRCh38.primary_assembly.genome.fa --extra_context 512 --embedding_idx 511 # autoreregressive
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_genalm_bert_large_t2t.csv genalm AIRI-Institute/gena-lm-bert-large-t2t ../GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_genalm_bigbird_base_t2t.csv genalm AIRI-Institute/gena-lm-bigbird-base-t2t ../GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_convnet.csv convnet pretrained_models/convnet data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_awd_lstm.csv awdlstm pretrained_models/awd_lstm data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 511 --extra_context 512 #autoreregressive
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_nt_v2_500m.csv nt InstaDeepAI/nucleotide-transformer-v2-500m-multi-species data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256
python3 scripts/predict_variant_effects.py $VARIANT_FILE $OUT_DIR/${OUT_PREFIX}_grover.csv grover pretrained_models/grover data/genomes/GRCh38.primary_assembly.genome.fa --embedding_idx 256