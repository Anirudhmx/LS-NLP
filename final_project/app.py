import streamlit as st
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

st.title("Text Completion bot")

if "message" not in st.session_state:
    st.session_state.message = []

for msg in st.session_state.message:
    st.chat_message(msg['role']).markdown(msg['content'])

model = GPT2LMHeadModel.from_pretrained("./finetuned_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./finetuned_gpt2")
# model.eval()

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.message.append({"role": "user", "content": prompt})

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 10,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.chat_message('assistant').markdown(generated_text)
    st.session_state.message.append({'role':'assistant', 'content':generated_text})





