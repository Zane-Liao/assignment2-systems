"""
---
Claude Sonnet4.5 Generate
---
"""
import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import sys

input_corpus_path = "../data/owt_train.txt"
vocab_size = 10000
output_dir = "vocab"
os.makedirs(output_dir, exist_ok=True)

output_vocab_path = os.path.join(output_dir, "vocab_train.txt")
output_merges_path = os.path.join(output_dir, "merges.txt")
output_tokenizer_path = os.path.join(output_dir, "tokenizer.json")

def pre_check():
    print("=" * 70)
    print("🔍 Pre-training Checks")
    print("=" * 70)
    
    if not os.path.exists(input_corpus_path):
        print(f"❌ ERROR: Input file not found: {input_corpus_path}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Please check the file path!")
        sys.exit(1)
    else:
        print(f"✅ Input file exists: {input_corpus_path}")
    
    file_size_gb = os.path.getsize(input_corpus_path) / (1024**3)
    print(f"✅ File size: {file_size_gb:.2f} GB")
    
    try:
        with open(input_corpus_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line.strip():
                print("⚠️  WARNING: First line is empty!")
            else:
                print(f"✅ First line preview: {first_line[:100]}")
    except Exception as e:
        print(f"❌ ERROR: Cannot read file: {e}")
        sys.exit(1)
    
    print("📊 Sampling file (counting ~1% of lines)...")
    try:
        with open(input_corpus_path, 'r', encoding='utf-8') as f:
            sample_lines = 0
            total_chars = 0
            for i, line in enumerate(f):
                if i >= 100000:
                    break
                if line.strip():
                    sample_lines += 1
                    total_chars += len(line)
        
        avg_chars = total_chars / sample_lines if sample_lines > 0 else 0
        print(f"✅ Sample: {sample_lines:,} lines, avg {avg_chars:.1f} chars/line")

        estimated_total_lines = int(file_size_gb * 1024**3 / avg_chars) if avg_chars > 0 else 0
        print(f"📈 Estimated total lines: ~{estimated_total_lines:,}")
        
    except Exception as e:
        print(f"⚠️  Sampling failed: {e}")
    
    print()

if __name__ == "__main__":
    pre_check()
    
    print("=" * 70)
    print("🚀 Begin Training BPE Tokenizer")
    print("=" * 70)
    print(f"📁 Corpus: {input_corpus_path}")
    print(f"🎯 Target vocab size: {vocab_size:,}")
    print()
    
    print("1️⃣  Initializing BPE model...")
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    print("   ✅ Model initialized")
    
    print("2️⃣  Setting pre-tokenizer...")

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    print("   ✅ Pre-tokenizer set (ByteLevel)")
    
    print("3️⃣  Setting up trainer...")
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=special_tokens,
        min_frequency=2
    )
    print(f"   ✅ Trainer configured")
    print(f"   Special tokens: {special_tokens}")
    
    print("\n4️⃣  Training tokenizer (this may take a while for 11GB file)...")
    print("   ⏳ Please wait...\n")
    
    try:
        tokenizer.train([input_corpus_path], trainer)
        print("\n   ✅ Training completed!")
    except Exception as e:
        print(f"\n   ❌ Training failed: {e}")
        sys.exit(1)

    print("   Setting decoder...")
    from tokenizers import decoders
    tokenizer.decoder = decoders.ByteLevel()
    print("   ✅ Decoder set")

    print("\n5️⃣  Validating vocabulary...")
    vocab = tokenizer.get_vocab()
    actual_vocab_size = len(vocab)
    print(f"   Target vocab size: {vocab_size:,}")
    print(f"   Actual vocab size: {actual_vocab_size:,}")
    
    if actual_vocab_size < 100:
        print(f"   ❌ ERROR: Vocab size too small ({actual_vocab_size})!")
        print(f"   This indicates training failed.")
        print(f"   Vocab content: {list(vocab.keys())[:20]}")
        sys.exit(1)
    elif actual_vocab_size < vocab_size * 0.9:
        print(f"   ⚠️  WARNING: Vocab smaller than expected")
        print(f"   This might be normal for small/repetitive datasets")
    else:
        print(f"   ✅ Vocab size is reasonable")

    print("\n6️⃣  Saving tokenizer...")
    tokenizer.save(output_tokenizer_path)
    print(f"   ✅ Saved to: {output_tokenizer_path}")
    
    print("7️⃣  Saving vocabulary...")
    with open(output_vocab_path, 'w', encoding='utf-8') as f:
        for token, id_ in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\t{id_}\n")
    print(f"   ✅ Saved to: {output_vocab_path} ({len(vocab):,} tokens)")
    
    print("8️⃣  Extracting and saving merges...")
    try:
        with open(output_tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        if 'model' in tokenizer_data and 'merges' in tokenizer_data['model']:
            merges_list = tokenizer_data['model']['merges']
            
            if merges_list:
                with open(output_merges_path, 'w', encoding='utf-8') as f:
                    f.write("#version: 0.2\n")
                    for merge_str in merges_list:
                        f.write(f"{merge_str}\n")
                
                print(f"   ✅ Saved to: {output_merges_path} ({len(merges_list):,} merges)")
            else:
                print("   ⚠️  No merges found (vocab might only contain base characters)")
        else:
            print("   ⚠️  No merges field in tokenizer.json")
    except Exception as e:
        print(f"   ⚠️  Failed to save merges: {e}")
    
    print("\n9️⃣  Testing tokenizer...")
    test_texts = [
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and natural language processing."
    ]
    
    print("   Test samples:")
    for i, text in enumerate(test_texts, 1):
        encoded = tokenizer.encode(text)
        print(f"\n   Test {i}: '{text}'")
        print(f"   Tokens ({len(encoded.tokens)}): {encoded.tokens[:15]}")
        print(f"   IDs: {encoded.ids[:15]}")
        
        decoded = tokenizer.decode(encoded.ids)
        if decoded.strip() == text.strip():
            print(f"   ✅ Decode OK")
        else:
            print(f"   ⚠️  Decode mismatch: '{decoded}'")
    
    print("\n" + "=" * 70)
    print("🎉 Tokenizer Training Complete!")
    print("=" * 70)
    print(f"📁 Output directory: {os.path.abspath(output_dir)}")
    print(f"📊 Vocabulary size: {actual_vocab_size:,}")
    print(f"📄 Files generated:")
    print(f"   - {output_tokenizer_path}")
    print(f"   - {output_vocab_path}")
    print(f"   - {output_merges_path}")
    print()