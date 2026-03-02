with open('bbpe_tokenizer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Replace line 55 (index 54) - fix the special_tokens list
lines[54] = '            special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "
