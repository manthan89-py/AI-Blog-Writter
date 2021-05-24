# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Import dependencies

# %%
from transformers import GPT2LMHeadModel , GPT2Tokenizer

# %% [markdown]
# ### Load Model

# %%
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large' , pad_token_id = tokenizer.eos_token_id )


# %%
tokenizer.decode(tokenizer.eos_token_id)

# %% [markdown]
# ### Tokenize the text

# %%
sentence = "Love is"
input_ids = tokenizer.encode(sentence , return_tensors = 'pt')


# %%
input_ids


# %%
tokenizer.decode(input_ids[0])


# %%
print(tokenizer.decode(input_ids[0][1]))
print(tokenizer.decode(input_ids[0][2]))
print(tokenizer.decode(input_ids[0][3]))
print(tokenizer.decode(input_ids[0][4]))

# %% [markdown]
# ### Generate and Decode Text

# %%
output = model.generate(input_ids, max_length = 500, num_beams = 5,no_repeat_ngram_size  = 2 , early_stopping = True)


# %%
output[0]


# %%
print(tokenizer.decode(output[0] , skip_special_tokens = True))

# %% [markdown]
# ### Output Result

# %%
text = tokenizer.decode(output[0] , skip_special_tokens = True)


# %%
with open('blog_ai.txt','w') as f:
    f.write(text)


