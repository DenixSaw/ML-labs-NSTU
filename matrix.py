from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


model = AutoModelForQuestionAnswering.from_pretrained("squirro/distilroberta-base-squad_v2")
tokenizer = AutoTokenizer.from_pretrained("squirro/distilroberta-base-squad_v2")
qa_model = pipeline("question-answering", model=model, tokenizer=tokenizer)

result_1 = qa_model(
    question="Who is Neo?",
    context="A cult sci-fi film about the hacker Neo, who discovers that reality is a simulation created by machines to enslave humanity. Under the guidance of Morpheus and Trinity, Neo joins the resistance, realizes he is the Chosen One, and learns to control the Matrix to free humanity.",
    handle_impossible_answer=True
)

result_2 = qa_model(
    question="What is Matrix?",
    context="A cult sci-fi film about the hacker Neo, who discovers that reality is a simulation created by machines to enslave humanity. Under the guidance of Morpheus and Trinity, Neo joins the resistance, realizes he is the Chosen One, and learns to control the Matrix to free humanity.",
    handle_impossible_answer=True
)

print(result_1)
print(result_2)