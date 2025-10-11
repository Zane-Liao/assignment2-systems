"""
---
Claude Sonnet4.5 Generate
---
"""
import os
import numpy as np
from tokenizers import Tokenizer
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import time
import mmap

tokenizer = None
eos_id = None

def init_worker(tokenizer_path, eos_token_id):
    global tokenizer, eos_id
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = eos_token_id

def encode_batch_worker(lines):
    global tokenizer, eos_id
    
    encodings = tokenizer.encode_batch(lines)
    
    all_ids = []
    for enc in encodings:
        all_ids.extend(enc.ids)
        if eos_id != -1:
            all_ids.append(eos_id)
    
    return np.array(all_ids, dtype=np.uint16)


class UltraFastEncoder:
    
    def __init__(self, tokenizer_path, num_workers=None):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        if self.eos_id is None:
            self.eos_id = -1
            print("⚠️  <EOS> not found, skipping EOS insertion")

        if num_workers is None:
            detected_cores = cpu_count()

            self.num_workers = min(8, max(1, detected_cores - 1))
        else:
            self.num_workers = min(num_workers, 16)

        print(f"🚀 Initialized with {self.num_workers} workers (detected {cpu_count()} cores)")
    
    def count_lines_fast(self, filepath):
        print("📊 Counting lines...")
        start = time.time()
        
        with open(filepath, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            lines = 0
            while mm.readline():
                lines += 1
            mm.close()
        
        elapsed = time.time() - start
        print(f"   Found {lines:,} lines in {elapsed:.2f}s")
        return lines
    
    def read_file_chunks(self, filepath, chunk_size):
        with open(filepath, 'r', encoding='utf-8') as f:
            chunk = []
            for line in f:
                line = line.strip()
                if line:
                    chunk.append(line)
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
            
            if chunk:
                yield chunk
    
    def encode_multiprocess(self, input_path, output_path, batch_size=50000):
        print(f"\n{'='*70}")
        print(f"🔥 Multi-process encoding: {input_path}")
        print(f"{'='*70}")
        
        file_size_gb = os.path.getsize(input_path) / (1024**3)
        print(f"📁 File size: {file_size_gb:.2f} GB")
        
        if file_size_gb > 5:
            print(f"⚠️  WARNING: Large file detected!")
            print(f"   For files > 5GB, consider using encode_hybrid() instead")
            print(f"   to avoid memory issues.")
            response = input("   Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("   Aborted. Use encoder.encode_hybrid() instead.")
                return 0
        
        start_time = time.time()

        total_lines = self.count_lines_fast(input_path)
        
        print(f"📖 Reading file into memory...")
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"   Loaded {len(lines):,} lines")
        
        batches = []
        for i in range(0, len(lines), batch_size):
            batches.append(lines[i:i + batch_size])
        
        print(f"   Split into {len(batches):,} batches")
        print(f"\n🔄 Encoding with {self.num_workers} workers...")
        
        with Pool(
            self.num_workers,
            initializer=init_worker,
            initargs=(self.tokenizer_path, self.eos_id)
        ) as pool:
            results = []
            completed = 0

            for result in pool.imap(encode_batch_worker, batches, chunksize=1):
                results.append(result)
                completed += 1
                
                if completed % 10 == 0 or completed == len(batches):
                    elapsed = time.time() - start_time
                    progress = completed / len(batches) * 100
                    speed = (completed * batch_size) / elapsed
                    total_tokens = sum(len(r) for r in results)
                    print(f"   [{completed}/{len(batches)}] {progress:.1f}% | "
                          f"{speed:.0f} lines/s | {total_tokens:,} tokens",
                          end='\r')
        
        print()

        print("🔗 Merging results...")
        arr = np.concatenate(results)
        

        print("💾 Saving to disk...")
        arr.tofile(output_path)
        

        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"\n{'='*70}")
        print(f"✅ Encoding complete!")
        print(f"{'='*70}")
        print(f"   Total tokens: {len(arr):,}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
        print(f"   Speed: {total_lines/elapsed:.0f} lines/s")
        print(f"   Avg tokens/line: {len(arr)/total_lines:.1f}")
        
        return len(arr)
    
    def encode_streaming(self, input_path, output_path, batch_size=50000):
        print(f"\n{'='*70}")
        print(f"🌊 Streaming encoding: {input_path}")
        print(f"{'='*70}")
        start_time = time.time()
        
        total_lines = self.count_lines_fast(input_path)
        
        print(f"🔄 Processing...")
        total_tokens = 0
        lines_processed = 0
        
        with open(output_path, "wb") as out_f:
            for batch in self.read_file_chunks(input_path, batch_size):

                encodings = self.tokenizer.encode_batch(batch)
                

                for enc in encodings:
                    ids = enc.ids
                    if self.eos_id != -1:
                        ids = ids + [self.eos_id]
                    
                    arr = np.array(ids, dtype=np.uint16)
                    arr.tofile(out_f)
                    total_tokens += len(ids)
                
                lines_processed += len(batch)
                

                elapsed = time.time() - start_time
                progress = lines_processed / total_lines * 100
                speed = lines_processed / elapsed
                print(f"   [{lines_processed:,}/{total_lines:,}] {progress:.1f}% | "
                      f"{speed:.0f} lines/s | {total_tokens:,} tokens",
                      end='\r')
        
        print()
        
        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"\n{'='*70}")
        print(f"✅ Encoding complete!")
        print(f"{'='*70}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
        print(f"   Speed: {total_lines/elapsed:.0f} lines/s")
        print(f"   Avg tokens/line: {total_tokens/total_lines:.1f}")
        
        return total_tokens
    
    def encode_hybrid(self, input_path, output_path, 
                     batch_size=50000, memory_limit_gb=8):
        print(f"\n{'='*70}")
        print(f"⚡ Hybrid encoding: {input_path}")
        print(f"   Memory limit: {memory_limit_gb} GB")
        print(f"{'='*70}")
        start_time = time.time()
        
        total_lines = self.count_lines_fast(input_path)
        
        lines_per_chunk = int(memory_limit_gb * 1024**3 / (100 * 4 * self.num_workers))
        lines_per_chunk = max(lines_per_chunk, batch_size * 10)
        
        print(f"   Processing in chunks of {lines_per_chunk:,} lines")
        
        total_tokens = 0
        lines_processed = 0
        
        with open(output_path, "wb") as out_f:
            chunk_lines = []
            
            with open(input_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    chunk_lines.append(line)
                    
                    if len(chunk_lines) >= lines_per_chunk:
                        tokens = self._process_chunk_multiprocess(
                            chunk_lines, batch_size, out_f
                        )
                        total_tokens += tokens
                        lines_processed += len(chunk_lines)

                        elapsed = time.time() - start_time
                        progress = lines_processed / total_lines * 100
                        speed = lines_processed / elapsed
                        print(f"   [{lines_processed:,}/{total_lines:,}] {progress:.1f}% | "
                              f"{speed:.0f} lines/s | {total_tokens:,} tokens")
                        
                        chunk_lines = []
                
                if chunk_lines:
                    tokens = self._process_chunk_multiprocess(
                        chunk_lines, batch_size, out_f
                    )
                    total_tokens += tokens
                    lines_processed += len(chunk_lines)
        
        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"\n{'='*70}")
        print(f"✅ Encoding complete!")
        print(f"{'='*70}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
        print(f"   Speed: {total_lines/elapsed:.0f} lines/s")
        print(f"   Avg tokens/line: {total_tokens/total_lines:.1f}")
        
        return total_tokens
    
    def _process_chunk_multiprocess(self, lines, batch_size, out_file):
        batches = []
        for i in range(0, len(lines), batch_size):
            batches.append(lines[i:i + batch_size])
        
        with Pool(
            self.num_workers,
            initializer=init_worker,
            initargs=(self.tokenizer_path, self.eos_id)
        ) as pool:
            results = pool.map(encode_batch_worker, batches)

        total_tokens = 0
        for arr in results:
            arr.tofile(out_file)
            total_tokens += len(arr)
        
        return total_tokens


if __name__ == "__main__":
    encoder = UltraFastEncoder(
        tokenizer_path="vocab/vocab_train.json",
        num_workers=None
    )
    
    print("\n" + "⚡" * 35)
    print("MODE: Hybrid (balanced speed & memory)")
    print("⚡" * 35)
    
    encoder.encode_hybrid(
        "../data/owt_train.txt",
        "../data/train.bin",
        batch_size=50000,
        memory_limit_gb=4
    )
    
    encoder.encode_hybrid(
        "../data/owt_valid.txt",
        "../data/valid.bin",
        batch_size=50000,
        memory_limit_gb=4
    )
    
    print("\n" + "🎉" * 35)
    print("All encoding tasks completed!")
    print("🎉" * 35)