from transformers import CLIPTokenizer

clip_tokenizer = CLIPTokenizer.from_pretrained("/mnt/data/wang_shaokun/CTVR/clip-vit-base-patch32",
                                               TOKENIZERS_PARALLELISM=False, clean_up_tokenization_spaces=True)
